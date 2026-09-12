import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from wfo.schema import IterationResults, WindowResult, write_results
from wfo.windows import walk_forward_windows

PREDICTION_HORIZON_MINUTES = 5
ROLLING_WINDOW_MINUTES = 20
FEATURE_COLS = ["rolling_mean_return", "rolling_vol", "volume_z"]
MIN_TRAIN_ROWS = 100
MIN_PREDICT_ROWS = 20


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

    future_close = slice_df["close"].shift(-PREDICTION_HORIZON_MINUTES)
    slice_df["label"] = (future_close > slice_df["close"]).astype("Int64")
    slice_df["future_return"] = future_close / slice_df["close"] - 1

    return slice_df.dropna(subset=[*FEATURE_COLS, "label", "future_return"])


def backtest(predict_df: pd.DataFrame, predictions, horizon_minutes: int) -> tuple[float, float, pd.DataFrame]:
    predict_df = predict_df.reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)

    # future_return looks `horizon_minutes` ahead, so consecutive rows overlap.
    # Compounding every row would count the same price move many times over.
    # Only take non-overlapping decision points, spaced by the horizon.
    trade_idx = list(range(0, len(predict_df), horizon_minutes))
    trades = predict_df.iloc[trade_idx].reset_index(drop=True)
    trade_predictions = predictions.iloc[trade_idx].reset_index(drop=True)

    strategy_returns = trade_predictions * trades["future_return"]
    equity = (1 + strategy_returns).cumprod()
    roi = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    ledger = trades[["datetime", "close", "future_return"]].copy()
    ledger["predicted"] = trade_predictions
    ledger["strategy_return"] = strategy_returns
    ledger["equity"] = equity
    return roi, max_drawdown, ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--train-months", type=int, required=True)
    parser.add_argument("--predict-months", type=int, required=True)
    parser.add_argument("--gap-days", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = load_ticker_data(args.data_dir, args.ticker)
    start, end = df["datetime"].min().date(), df["datetime"].max().date()
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
        roi, max_drawdown, ledger = backtest(predict, predictions, PREDICTION_HORIZON_MINUTES)

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
                max_drawdown=max_drawdown,
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
