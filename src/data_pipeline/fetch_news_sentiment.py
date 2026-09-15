import csv
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient

load_dotenv()

TICKERS = ["AAPL", "MSFT", "SPY", "NVDA", "AMZN", "GOOGL", "META", "JPM"]
YEARS_BACK = 2
OUTPUT_DIR = Path("data/news")
SECONDS_BETWEEN_CALLS = 13
SECONDS_BETWEEN_PAGES = 20
PAGE_SIZE = 1000
MAX_RETRIES = 8
CSV_HEADER = ["published_utc", "sentiment_score", "title"]
SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def last_published_utc(path: Path) -> str | None:
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
    return lines[-1].split(",")[0]


def article_sentiment(article, ticker: str) -> float | None:
    scores = [
        SENTIMENT_SCORE[i.sentiment]
        for i in (article.insights or [])
        if i.ticker == ticker and i.sentiment in SENTIMENT_SCORE
    ]
    return sum(scores) / len(scores) if scores else None


def fetch_ticker(client: RESTClient, ticker: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{ticker}.csv"
    since = last_published_utc(path)

    if since is None:
        print(f"[{ticker}] no existing news, backfilling {YEARS_BACK} years")
        cursor = (date.today() - timedelta(days=365 * YEARS_BACK)).isoformat()
        inclusive, write_header = True, True
    else:
        print(f"[{ticker}] resuming from {since}")
        cursor, inclusive, write_header = since, False, False

    mode = "w" if write_header else "a"
    f = path.open(mode, newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(CSV_HEADER)

    total = 0
    attempt = 0
    try:
        while True:
            attempt += 1
            try:
                kwargs = dict(ticker=ticker, limit=PAGE_SIZE, sort="published_utc", order="asc")
                kwargs["published_utc_gte" if inclusive else "published_utc_gt"] = cursor
                seen_this_attempt = 0
                for article in client.list_ticker_news(**kwargs):
                    cursor = article.published_utc
                    inclusive = False
                    score = article_sentiment(article, ticker)
                    if score is not None:
                        writer.writerow([article.published_utc, score, (article.title or "").replace(",", ";")])
                        total += 1
                    seen_this_attempt += 1
                    if seen_this_attempt % PAGE_SIZE == 0:
                        f.flush()
                        time.sleep(SECONDS_BETWEEN_PAGES)
                break
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    raise
                wait = SECONDS_BETWEEN_CALLS * attempt
                print(f"[{ticker}] error ({e}), retrying from {cursor} in {wait}s")
                f.flush()
                time.sleep(wait)
    finally:
        f.close()

    print(f"[{ticker}] wrote {total} new sentiment-scored articles")


def main():
    client = RESTClient(os.environ["MASSIVE_API_KEY"])
    for i, ticker in enumerate(TICKERS):
        fetch_ticker(client, ticker)
        if i < len(TICKERS) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)


if __name__ == "__main__":
    main()
