import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from wfo.schema import IterationResults, WindowResult, write_results
from wfo.windows import walk_forward_windows

# ---------------------------------------------------------------------------
# DAY-TRADE (open-to-close) reference model. The decision is made once per
# day AT THE OPEN, and the position is always closed AT THE CLOSE of the
# SAME day — never held overnight. This introduces a leakage risk distinct
# from both the intraday-minute and multi-day-swing models:
#
# Today's own high/low/close/volume/vwap are NOT known at the moment you'd
# decide to enter at today's open — they only exist once the day is over.
# Using them as FEATURES would leak the day's own outcome into the decision.
# Only two things are legitimately knowable at today's open:
#   1. Everything through YESTERDAY's close (all standard indicators,
#      computed normally then shifted forward by 1 row)
#   2. TODAY's own open price itself (e.g. the gap vs yesterday's close)
#
# The LABEL (today's own open-to-close return) is fine to compute directly
# from today's open/close — it's the training TARGET, never a feature.
#
# Because the holding period exactly equals one row (open to close, same
# day), there is no overlap between consecutive trades — every row is an
# independent, non-overlapping decision point. No striding/spacing logic
# is needed here (unlike swing or intraday-minute models).
# ---------------------------------------------------------------------------

LABEL_RETURN_THRESHOLD = 0.0   # today counts as "up" if close > open at all
RSI_PERIOD = 14
SMA_FAST, SMA_SLOW = 5, 15
ATR_PERIOD = 14
VOL_WINDOW = 20

FEATURE_COLS = ["rsi", "sma_cross", "roc_5", "atr_pct", "volume_z", "gap_pct", "prior_range_pct"]

MIN_TRAIN_ROWS = 150
MIN_PREDICT_ROWS = 15
STARTING_CAPITAL_USD = 10_000
COMMISSION_PER_TRADE_USD = 1.0
LONG_ONLY = False


def load_ticker_data(data_dir: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{ticker}.csv")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, np.nan)
    return 100 - (100 / (1 + gain / loss))


def prepare_window_slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    mask = (df["datetime"].dt.date >= start) & (df["datetime"].dt.date < end)
    d = df.loc[mask].reset_index(drop=True).copy()
    if len(d) == 0:
        return d

    close = d["close"]
    prev_close = close.shift(1)

    # All of these reflect information available as of YESTERDAY's close —
    # computed normally then shifted 1 row so row T sees T-1's value, never
    # T's own (which wouldn't exist yet at T's open).
    sma_fast = close.rolling(SMA_FAST).mean()
    sma_slow = close.rolling(SMA_SLOW).mean()
    d["sma_cross"] = ((sma_fast - sma_slow) / sma_slow).shift(1)
    d["rsi"] = _rsi(close, RSI_PERIOD).shift(1)
    d["roc_5"] = close.pct_change(5).shift(1)

    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    d["atr_pct"] = (tr.rolling(ATR_PERIOD).mean() / close).shift(1)

    vol_mean = d["volume"].rolling(VOL_WINDOW).mean()
    vol_std = d["volume"].rolling(VOL_WINDOW).std()
    d["volume_z"] = ((d["volume"] - vol_mean) / vol_std.replace(0, np.nan)).shift(1)

    # Legitimately knowable AT today's open: the gap vs yesterday's close,
    # and yesterday's own high-low range (both use only T-1/T-open data).
    d["gap_pct"] = d["open"] / prev_close - 1
    d["prior_range_pct"] = (d["high"].shift(1) - d["low"].shift(1)) / prev_close

    # LABEL: today's own open-to-close return. Fine as a target — never used
    # as a feature above.
    d["future_return"] = d["close"] / d["open"] - 1
    d["label"] = (d["future_return"] > LABEL_RETURN_THRESHOLD).astype("Int64")

    return d.dropna(subset=[*FEATURE_COLS, "label", "future_return"]).reset_index(drop=True)


def buy_hold_roi(df: pd.DataFrame, start, end) -> float:
    mask = (df["datetime"].dt.date >= start) & (df["datetime"].dt.date < end)
    raw = df.loc[mask]
    if len(raw) < 2:
        return 0.0
    return float(raw["close"].iloc[-1] / raw["close"].iloc[0] - 1)


def backtest(
    predict_df: pd.DataFrame, predictions, long_only: bool = LONG_ONLY
) -> tuple[float, float, float, int, pd.DataFrame]:
    predict_df = predict_df.reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)
    if long_only:
        predictions = predictions.clip(lower=0)
    num_trades = int((predictions != 0).sum())

    # Every row is an independent, non-overlapping open-to-close trade — no
    # striding needed (unlike swing/intraday, holding period == 1 row).
    strategy_returns = predictions * predict_df["future_return"]
    equity = (1 + strategy_returns).cumprod()
    roi = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    capital = STARTING_CAPITAL_USD
    net_equity = []
    for r, pred in zip(strategy_returns, predictions):
        capital = capital * (1 + r) - (COMMISSION_PER_TRADE_USD if pred != 0 else 0.0)
        net_equity.append(capital)
    net_roi = float(capital / STARTING_CAPITAL_USD - 1) if len(strategy_returns) else 0.0

    ledger = predict_df[["datetime", "open", "close", "future_return"]].copy()
    ledger["predicted"] = predictions
    ledger["strategy_return"] = strategy_returns
    ledger["equity"] = equity
    ledger["net_equity_usd"] = net_equity
    return roi, net_roi, max_drawdown, num_trades, ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--train-months", type=int, required=True)
    parser.add_argument("--predict-months", type=int, required=True)
    parser.add_argument("--gap-days", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-months", type=int, default=0)
    args = parser.parse_args()

    df = load_ticker_data(args.data_dir, args.ticker)
    start, full_end = df["datetime"].min().date(), df["datetime"].max().date()
    end = full_end - timedelta(days=30 * args.holdout_months)
    windows = walk_forward_windows(start, end, args.train_months, args.predict_months, args.gap_days)

    window_results = []
    for w in windows:
        train = prepare_window_slice(df, w.train_start, w.train_end)
        predict = prepare_window_slice(df, w.predict_start, w.predict_end)
        if len(train) < MIN_TRAIN_ROWS or len(predict) < MIN_PREDICT_ROWS:
            continue
        if train["label"].nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model.fit(train[FEATURE_COLS], train["label"])
        predictions = model.predict(predict[FEATURE_COLS])

        f1 = float(f1_score(predict["label"], predictions, zero_division=0))
        accuracy = float(accuracy_score(predict["label"], predictions))
        roi, net_roi, max_drawdown, num_trades, ledger = backtest(predict, predictions)
        bh_roi = buy_hold_roi(df, w.predict_start, w.predict_end)

        weights_path = args.output_dir / f"window_{w.index}_weights.pkl"
        ledger_path = args.output_dir / f"window_{w.index}_ledger.csv"
        weights_path.write_bytes(pickle.dumps(model))
        ledger.to_csv(ledger_path, index=False)

        window_results.append(
            WindowResult(
                index=w.index,
                train_start=w.train_start.isoformat(),
                train_end=w.train_end.isoformat(),
                predict_start=w.predict_start.isoformat(),
                predict_end=w.predict_end.isoformat(),
                f1=f1,
                accuracy=accuracy,
                roi=roi,
                net_roi=net_roi,
                max_drawdown=max_drawdown,
                buy_hold_roi=bh_roi,
                num_trades=num_trades,
                weights_path=str(weights_path),
                ledger_path=str(ledger_path),
            )
        )

    results = IterationResults(
        ticker=args.ticker,
        prediction_target="close_above_open_same_day",
        approach=(
            "RandomForestClassifier on prior-day SMA-crossover/RSI/ROC/ATR%/volume-zscore "
            "plus today's open gap and prior-day range, predicting whether today's close "
            "will exceed today's open; enter at open, exit at close, same day."
        ),
        train_months=args.train_months,
        predict_months=args.predict_months,
        gap_days=args.gap_days,
        windows=window_results,
    )
    write_results(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
