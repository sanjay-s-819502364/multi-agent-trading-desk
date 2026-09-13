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
