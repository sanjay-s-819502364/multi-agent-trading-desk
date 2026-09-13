import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from wfo.schema import IterationResults, WindowResult, write_results
from wfo.timeutils import time_based_future_return, time_based_trade_points
from wfo.windows import walk_forward_windows

PREDICTION_HORIZON_MINUTES = 5
ROLLING_WINDOW_MINUTES = 20
FEATURE_COLS = ["rolling_mean_return", "rolling_vol", "volume_z"]
MIN_TRAIN_ROWS = 100
MIN_PREDICT_ROWS = 20
STARTING_CAPITAL_USD = 10_000
COMMISSION_PER_TRADE_USD = 1.0
LONG_ONLY = False


def load_ticker_data(data_dir: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{ticker}.csv")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def prepare_window_slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    mask = (df["datetime"].dt.date >= start) & (df["datetime"].dt.date < end)
    slice_df = df.loc[mask].reset_index(drop=True).copy()

    slice_df["return_1m"] = slice_df["close"].pct_change()
    slice_df["rolling_mean_return"] = slice_df["return_1m"].rolling(ROLLING_WINDOW_MINUTES).mean()
    slice_df["rolling_vol"] = slice_df["return_1m"].rolling(ROLLING_WINDOW_MINUTES).std()
    slice_df["volume_z"] = (
        slice_df["volume"] - slice_df["volume"].rolling(ROLLING_WINDOW_MINUTES).mean()
    ) / slice_df["volume"].rolling(ROLLING_WINDOW_MINUTES).std()

    slice_df["future_return"] = time_based_future_return(slice_df, PREDICTION_HORIZON_MINUTES)
    slice_df["label"] = (slice_df["future_return"] > 0).astype("Int64")

    return slice_df.dropna(subset=[*FEATURE_COLS, "label", "future_return"])


def buy_hold_roi(df: pd.DataFrame, start, end) -> float:
    mask = (df["datetime"].dt.date >= start) & (df["datetime"].dt.date < end)
    raw = df.loc[mask]
    if len(raw) < 2:
        return 0.0
    return float(raw["close"].iloc[-1] / raw["close"].iloc[0] - 1)


def backtest(
    predict_df: pd.DataFrame, predictions, horizon_minutes: int, long_only: bool = LONG_ONLY
) -> tuple[float, float, float, int, pd.DataFrame]:
    predict_df = predict_df.reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)

    # future_return looks `horizon_minutes` ahead, so consecutive rows overlap.
    # Compounding every row would count the same price move many times over.
    # Only take non-overlapping decision points, spaced by real elapsed time
    # (not row count, since gaps in the data would otherwise let a "trade"
    # silently span hours or days instead of the intended horizon).
    trade_idx = time_based_trade_points(predict_df, horizon_minutes)
    trades = predict_df.iloc[trade_idx].reset_index(drop=True)
    trade_predictions = predictions.iloc[trade_idx].reset_index(drop=True)
    if long_only:
        trade_predictions = trade_predictions.clip(lower=0)
    num_trades = int((trade_predictions != 0).sum())

    strategy_returns = trade_predictions * trades["future_return"]
    equity = (1 + strategy_returns).cumprod()
    roi = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    # Net of a flat $1 brokerage fee per trade, against an assumed
    # STARTING_CAPITAL_USD account. A flat dollar fee is only meaningful as a
    # percentage relative to an assumed capital base, unlike the rest of this
    # backtest which works in pure returns from a notional 1.0 starting point.
    # Only charged when a trade is actually taken (nonzero position) — a flat
    # decision point isn't a transaction.
    capital = STARTING_CAPITAL_USD
    net_equity = []
    for r, pred in zip(strategy_returns, trade_predictions):
        capital = capital * (1 + r) - (COMMISSION_PER_TRADE_USD if pred != 0 else 0.0)
        net_equity.append(capital)
    net_roi = float(capital / STARTING_CAPITAL_USD - 1) if len(strategy_returns) else 0.0

    ledger = trades[["datetime", "close", "future_return"]].copy()
    ledger["predicted"] = trade_predictions
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
    parser.add_argument(
        "--holdout-months", type=int, default=0,
        help="Reserve this many trailing months of data from all WFO windows, so a later "
             "run with --holdout-months 0 extends into a genuinely unseen blind-test period.",
    )
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

        model = LogisticRegression(max_iter=1000)
        model.fit(train[FEATURE_COLS], train["label"])
        predictions = model.predict(predict[FEATURE_COLS])

        f1 = float(f1_score(predict["label"], predictions, zero_division=0))
        accuracy = float(accuracy_score(predict["label"], predictions))
        roi, net_roi, max_drawdown, num_trades, ledger = backtest(predict, predictions, PREDICTION_HORIZON_MINUTES)
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
        prediction_target=f"price_up_in_{PREDICTION_HORIZON_MINUTES}min",
        approach=(
            f"LogisticRegression on rolling_mean_return/rolling_vol/volume_z "
            f"features ({ROLLING_WINDOW_MINUTES}min window)"
        ),
        train_months=args.train_months,
        predict_months=args.predict_months,
        gap_days=args.gap_days,
        windows=window_results,
    )
    write_results(args.output_dir / "results.json", results)


if __name__ == "__main__":
    main()
