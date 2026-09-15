import numpy as np
import pandas as pd


def time_based_future_return(df: pd.DataFrame, horizon_minutes: int, tolerance_minutes: float = 1.0) -> pd.Series:
    """Forward return over `horizon_minutes` of real elapsed time, not row count.

    Minute-bar data has gaps (illiquid minutes, and multi-hour/overnight/weekend
    session boundaries). A positional `.shift(-N)` silently spans however much
    real time N rows happen to cover, which can be a weekend instead of N
    minutes. This looks up, for each row, the first future row whose timestamp
    is >= horizon_minutes later (within `tolerance_minutes` slack for ordinary
    bar-availability jitter), and returns NaN where no such row exists close
    enough to that target time — e.g. the horizon crossed a session boundary —
    rather than silently grabbing whatever bar comes next, hours or days later.
    """
    df = df.reset_index(drop=True)
    target = (df["datetime"] + pd.Timedelta(minutes=horizon_minutes)).astype(df["datetime"].dtype)
    lookup = df[["datetime", "close"]].rename(columns={"close": "future_close"})
    query = pd.DataFrame({"datetime": target})
    matched = pd.merge_asof(
        query,
        lookup,
        on="datetime",
        direction="forward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    future_close = matched["future_close"].to_numpy()
    return pd.Series(future_close / df["close"].to_numpy() - 1, index=df.index)


def attach_sentiment_features(
    bars: pd.DataFrame, news: pd.DataFrame, lookback_hours: float = 24.0
) -> pd.DataFrame:
    """Attach trailing news-sentiment features to each bar, leakage-safe.

    For each bar, `sentiment_mean`/`sentiment_count` summarize only articles
    published at or before that bar's own timestamp (never after) — computed
    by rolling the sentiment series over `lookback_hours` evaluated at each
    article's own publish time, then merge_asof'd backward onto the bars. A
    bar with no qualifying prior article gets a neutral 0.0/0 rather than NaN,
    since "no news yet" is itself a legitimate, known state at decision time.
    """
    bars = bars.reset_index(drop=True)
    if news.empty:
        bars["sentiment_mean"] = 0.0
        bars["sentiment_count"] = 0
        return bars

    news = news.sort_values("published_utc").reset_index(drop=True)
    rolled = (
        news.set_index("published_utc")["sentiment_score"]
        .rolling(pd.Timedelta(hours=lookback_hours))
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "sentiment_mean", "count": "sentiment_count"})
    )
    rolled["published_utc"] = rolled["published_utc"].astype(bars["datetime"].dtype)

    merged = pd.merge_asof(
        bars, rolled, left_on="datetime", right_on="published_utc", direction="backward",
    )
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["sentiment_count"] = merged["sentiment_count"].fillna(0).astype(int)
    return merged.drop(columns=["published_utc"])


def time_based_trade_points(df: pd.DataFrame, horizon_minutes: int) -> list[int]:
    """Positional indices for non-overlapping backtest decision points, spaced
    by at least `horizon_minutes` of real elapsed time, not row count.

    Row-count striding (e.g. `df.iloc[::horizon_minutes]`) produces wildly
    uneven real spacing when the data has gaps — a "trade" can silently span
    hours or days instead of the intended horizon, corrupting compounded ROI.
    """
    timestamps = df["datetime"].to_numpy()
    indices = []
    next_allowed = None
    for i, ts in enumerate(timestamps):
        if next_allowed is None or ts >= next_allowed:
            indices.append(i)
            next_allowed = ts + np.timedelta64(horizon_minutes, "m")
    return indices
