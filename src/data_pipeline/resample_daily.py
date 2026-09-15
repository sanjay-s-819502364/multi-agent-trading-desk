from pathlib import Path

import pandas as pd

MINUTE_DIR = Path("data/minute_aggs")
DAILY_DIR = Path("data/daily_aggs")


def resample_to_daily(minute_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(minute_csv)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date

    dollar_volume = df["vwap"] * df["volume"]

    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        transactions=("transactions", "sum"),
    )
    daily["vwap"] = dollar_volume.groupby(df["date"]).sum() / daily["volume"]
    daily = daily.reset_index()

    daily["timestamp"] = pd.to_datetime(daily["date"]).astype("int64") * 1000
    return daily[["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]]


def main():
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    for minute_csv in sorted(MINUTE_DIR.glob("*.csv")):
        ticker = minute_csv.stem
        daily = resample_to_daily(minute_csv)
        out_path = DAILY_DIR / f"{ticker}.csv"
        daily.to_csv(out_path, index=False)
        print(f"[{ticker}] {len(daily)} daily bars -> {out_path}")


if __name__ == "__main__":
    main()
