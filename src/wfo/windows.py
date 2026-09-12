from dataclasses import dataclass
from datetime import date, timedelta

MONTH_DAYS = 30


@dataclass(frozen=True)
class Window:
    index: int
    train_start: date
    train_end: date
    predict_start: date
    predict_end: date


def walk_forward_windows(
    start: date,
    end: date,
    train_months: int,
    predict_months: int,
    gap_days: int,
) -> list[Window]:
    if gap_days < 1:
        raise ValueError("gap_days must be at least 1")

    train_span = train_months * MONTH_DAYS
    predict_span = predict_months * MONTH_DAYS

    windows = []
    train_start = start
    index = 0
    while True:
        train_end = _add_days(train_start, train_span)
        predict_start = _add_days(train_end, gap_days)
        predict_end = _add_days(predict_start, predict_span)
        if predict_end > end:
            break
        windows.append(Window(index, train_start, train_end, predict_start, predict_end))
        train_start = _add_days(train_start, predict_span)
        index += 1
    return windows


def _add_days(d: date, days: int) -> date:
    return d + timedelta(days=days)
