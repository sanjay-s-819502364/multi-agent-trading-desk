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
# OVERNIGHT reference model. The decision is made AT TODAY's CLOSE, the
# position is held overnight, and closed AT TOMORROW's OPEN. This is the
# mirror image of the day-trade (open-to-close) model, and is grounded in a
# real, documented market anomaly: a large share of long-run U.S. equity
# returns accrues overnight (close-to-open), not during the trading session.
#
# Because the decision happens AFTER today's close, today's own full OHLCV
# is legitimately known and safe to use as features directly — no shift(1)
# trick needed (unlike the day-trade model, where today's own bar isn't
# known yet at the decision point). The only thing requiring a forward look
# is the LABEL itself (tomorrow's open), which is fine since it's the
# training target, never a feature.
#
# Each overnight trade (close[T] -> open[T+1]) is independent and does not
# overlap the next one, so — like the day-trade model — no striding/spacing
# logic is needed in the backtest; every row is a valid decision point.
# ---------------------------------------------------------------------------

LABEL_RETURN_THRESHOLD = 0.0   # tonight counts as "up" if next open > today's close at all
RSI_PERIOD = 14
SMA_FAST, SMA_SLOW = 5, 15
ATR_PERIOD = 14
VOL_WINDOW = 20

FEATURE_COLS = ["rsi", "sma_cross", "roc_5", "atr_pct", "volume_z", "today_return", "close_vs_vwap"]

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

    # Today's own full bar is known at the decision point (made at today's
    # close) — no shift needed for these, unlike the day-trade model.
    sma_fast = close.rolling(SMA_FAST).mean()
    sma_slow = close.rolling(SMA_SLOW).mean()
    d["sma_cross"] = (sma_fast - sma_slow) / sma_slow
    d["rsi"] = _rsi(close, RSI_PERIOD)
    d["roc_5"] = close.pct_change(5)

    prev_close = close.shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    d["atr_pct"] = tr.rolling(ATR_PERIOD).mean() / close

    vol_mean = d["volume"].rolling(VOL_WINDOW).mean()
    vol_std = d["volume"].rolling(VOL_WINDOW).std()
    d["volume_z"] = (d["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    d["today_return"] = close / d["open"] - 1
    d["close_vs_vwap"] = close / d["vwap"] - 1

    # LABEL: tonight's overnight return, tomorrow's open vs today's close.
    # This is the ONLY forward-looking piece — a label, never a feature.
    next_open = d["open"].shift(-1)
    d["future_return"] = next_open / close - 1
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

    # Independent, non-overlapping overnight trades — every row valid, no
    # striding needed (same reasoning as the day-trade model).
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

    ledger = predict_df[["datetime", "close", "future_return"]].copy()
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
        prediction_target="next_open_above_today_close_overnight",
        approach=(
            "RandomForestClassifier on today's RSI/SMA-crossover/ROC/ATR%/volume-zscore/"
            "own-day-return/close-vs-vwap, predicting whether tomorrow's open will exceed "
            "today's close; buy at close, sell at next open, holding overnight."
        ),
        train_months=args.train_months,
        predict_months=args.predict_months,
        gap_days=args.gap_days,
        windows=window_results,
    )
    write_results(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
