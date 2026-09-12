import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindowResult:
    index: int
    train_start: str
    train_end: str
    predict_start: str
    predict_end: str
    f1: float
    accuracy: float
    roi: float
    max_drawdown: float
    weights_path: str
    ledger_path: str


@dataclass(frozen=True)
class IterationResults:
    ticker: str
    prediction_target: str
    approach: str
    train_months: int
    predict_months: int
    gap_days: int
    windows: list[WindowResult]


REQUIRED_WINDOW_FIELDS = set(WindowResult.__dataclass_fields__)
REQUIRED_TOP_FIELDS = set(IterationResults.__dataclass_fields__)


def write_results(path: Path, results: IterationResults) -> None:
    payload = asdict(results)
    path.write_text(json.dumps(payload, indent=2))


def load_results(path: Path) -> IterationResults:
    payload = json.loads(path.read_text())
    missing_top = REQUIRED_TOP_FIELDS - set(payload)
    if missing_top:
        raise ValueError(f"results.json missing fields: {sorted(missing_top)}")

    windows = []
    for w in payload["windows"]:
        missing = REQUIRED_WINDOW_FIELDS - set(w)
        if missing:
            raise ValueError(f"window entry missing fields: {sorted(missing)}")
        windows.append(WindowResult(**w))

    return IterationResults(
        ticker=payload["ticker"],
        prediction_target=payload["prediction_target"],
        approach=payload["approach"],
        train_months=payload["train_months"],
        predict_months=payload["predict_months"],
        gap_days=payload["gap_days"],
        windows=windows,
    )
