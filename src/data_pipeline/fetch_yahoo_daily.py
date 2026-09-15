import time
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "SPY", "NVDA", "AMZN", "GOOGL", "META", "JPM"]
YEARS_BACK = "10y"
OUTPUT_DIR = Path("data/daily_aggs")
SECONDS_BETWEEN_CALLS = 3


def fetch_ticker(ticker: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=YEARS_BACK, interval="1d", auto_adjust=False)
    hist = hist.reset_index()
    dates_utc = hist["Date"].dt.tz_convert("UTC") if hist["Date"].dt.tz is not None else hist["Date"].dt.tz_localize("UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    hist["timestamp"] = ((dates_utc - epoch).dt.total_seconds() * 1000).astype("int64")
    hist["vwap"] = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    hist["transactions"] = 0
    return hist.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })[["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, ticker in enumerate(TICKERS):
        df = fetch_ticker(ticker)
        out_path = OUTPUT_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"[{ticker}] {len(df)} daily bars ({YEARS_BACK}) -> {out_path}")
        if i < len(TICKERS) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
