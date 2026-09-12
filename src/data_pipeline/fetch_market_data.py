import csv
import os
import time
from datetime import date, datetime, timedelta, timezone
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
CSV_HEADER = ["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]


def date_windows(start: date, end: date):
    windows = []
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(days=30), end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def last_timestamp(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = min(size, 4096)
        f.seek(size - block)
        chunk = f.read(block).decode("utf-8", errors="ignore")
    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines or lines[-1] == ",".join(CSV_HEADER):
        return None
    return int(lines[-1].split(",")[0])


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


def write_bars(ticker: str, bars, write_header: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{ticker}.csv"
    mode = "w" if write_header else "a"
    with path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for bar in bars:
            writer.writerow(
                [bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap, bar.transactions]
            )


def fetch_ticker(client: RESTClient, ticker: str):
    path = OUTPUT_DIR / f"{ticker}.csv"
    since_ts = last_timestamp(path)
    today = date.today()

    if since_ts is None:
        print(f"[{ticker}] no existing data, backfilling {YEARS_BACK} years")
        start = today - timedelta(days=365 * YEARS_BACK)
        write_header = True
    else:
        start = datetime.fromtimestamp(since_ts / 1000, tz=timezone.utc).date()
        print(f"[{ticker}] resuming from {start.isoformat()}")
        write_header = False

    if start >= today:
        print(f"[{ticker}] already up to date")
        return

    windows = date_windows(start, today)
    for i, (window_start, window_end) in enumerate(windows):
        bars = fetch_window(client, ticker, window_start, window_end)
        if since_ts is not None:
            bars = [b for b in bars if b.timestamp > since_ts]
        write_bars(ticker, bars, write_header=(write_header and i == 0))
        print(f"[{ticker}] {window_start.isoformat()} to {window_end.isoformat()}: {len(bars)} bars")
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
