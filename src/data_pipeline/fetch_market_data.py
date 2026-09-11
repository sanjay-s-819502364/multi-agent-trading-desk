import csv
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient
from massive.exceptions import BadResponse

load_dotenv()

TICKERS = ["AAPL", "MSFT", "SPY", "NVDA", "AMZN"]
YEARS_BACK = 2
OUTPUT_DIR = Path("data/minute_aggs")
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 3


def month_windows(years_back: int):
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    windows = []
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=30), end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def fetch_window(client: RESTClient, ticker: str, start: date, end: date):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=start.isoformat(),
                to=end.isoformat(),
                limit=50000,
            )
        except BadResponse as e:
            if "doesn't include this data timeframe" in str(e):
                return []
            if attempt == MAX_RETRIES:
                raise
            time.sleep(SECONDS_BETWEEN_CALLS * attempt)
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(SECONDS_BETWEEN_CALLS * attempt)


def write_bars(ticker: str, bars, append: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{ticker}.csv"
    mode = "a" if append else "w"
    with path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(
                ["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]
            )
        for bar in bars:
            writer.writerow(
                [bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap, bar.transactions]
            )


def fetch_ticker(client: RESTClient, ticker: str):
    windows = month_windows(YEARS_BACK)
    for i, (start, end) in enumerate(windows):
        bars = fetch_window(client, ticker, start, end)
        write_bars(ticker, bars, append=(i > 0))
        if i < len(windows) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)


def main():
    client = RESTClient(os.environ["MASSIVE_API_KEY"])
    for i, ticker in enumerate(TICKERS):
        fetch_ticker(client, ticker)
        if i < len(TICKERS) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
