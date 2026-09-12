import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

from agents.model_designer import generate_model_script
from wfo.run_iteration import ITERATIONS_DIR, run_iteration
from wfo.schema import IterationResults, load_results

MAX_ITERATIONS = 5
ROI_ACCEPT_THRESHOLD = 0.20
DEFAULT_TRAIN_MONTHS = 3
DEFAULT_PREDICT_MONTHS = 1
DEFAULT_GAP_DAYS = 2


def diagnose(results: IterationResults) -> dict:
    rois = [w.roi for w in results.windows]
    f1s = [w.f1 for w in results.windows]
    collapsed = sum(1 for w in results.windows if w.f1 == 0.0)
    return {
        "avg_roi": statistics.fmean(rois),
        "median_roi": statistics.median(rois),
        "avg_f1": statistics.fmean(f1s),
        "num_windows": len(results.windows),
        "collapsed_windows": collapsed,
    }


def build_instructions(prev_approach: str | None, prev_diag: dict | None) -> str:
    if prev_diag is None:
        return (
            "Design an initial model for intraday minute-bar signal prediction. "
            "Pick your own feature set, model, and prediction framing."
        )

    parts = [
        f"Previous iteration ({prev_approach}) scored avg ROI={prev_diag['avg_roi']:+.3f}, "
        f"avg F1={prev_diag['avg_f1']:.3f}, with {prev_diag['collapsed_windows']} of "
        f"{prev_diag['num_windows']} windows collapsing to a constant majority-class prediction."
    ]
    if prev_diag["collapsed_windows"] > prev_diag["num_windows"] / 2:
        parts.append(
            "More than half the windows collapsed to a constant prediction — address class "
            "imbalance or add stronger discriminative features rather than just retuning."
        )
    parts.append(
        "Improve on this: aim for higher and more consistent ROI across windows. Try a "
        "different feature set, model type, or prediction framing if this approach has plateaued."
    )
    return " ".join(parts)


def print_iteration_report(iteration_num: int, results: IterationResults, diag: dict, prev_diag: dict | None):
    print(f"\n[iteration {iteration_num}/{MAX_ITERATIONS}]")
    print(f"  approach: {results.approach}")
    print(f"  prediction target: {results.prediction_target}")
    if prev_diag is not None:
        print(f"  avg ROI: {prev_diag['avg_roi']:+.3f} -> {diag['avg_roi']:+.3f}")
        print(f"  avg F1:  {prev_diag['avg_f1']:.3f} -> {diag['avg_f1']:.3f}")
    else:
        print(f"  avg ROI: {diag['avg_roi']:+.3f}")
        print(f"  avg F1:  {diag['avg_f1']:.3f}")
    print(f"  windows: {diag['num_windows']} ({diag['collapsed_windows']} collapsed to constant prediction)")
    for w in results.windows:
        print(f"    window {w.index:2d}  f1={w.f1:.3f}  acc={w.accuracy:.3f}  roi={w.roi:+.3f}  maxdd={w.max_drawdown:+.3f}")


def run_loop(ticker: str, train_months: int, predict_months: int, gap_days: int) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    approach = None
    diag = None

    for i in range(1, MAX_ITERATIONS + 1):
        instructions = build_instructions(approach, diag)
        tag = f"{run_id}_{ticker}_iter{i}"
        script_path = generate_model_script(instructions, iteration_tag=tag)

        ok, outcome = run_iteration(
            script_path, ticker, train_months, predict_months, gap_days, iteration_id=tag
        )
        if not ok:
            print(f"\n[iteration {i}/{MAX_ITERATIONS}] FAILED: {outcome}")
            continue

        results = load_results(ITERATIONS_DIR / outcome / "results.json")
        new_diag = diagnose(results)
        print_iteration_report(i, results, new_diag, diag)

        approach, diag = results.approach, new_diag

        if new_diag["avg_roi"] >= ROI_ACCEPT_THRESHOLD:
            print(f"\nACCEPTED at iteration {i}: avg ROI {new_diag['avg_roi']:.3f} >= threshold {ROI_ACCEPT_THRESHOLD}")
            return

    print(f"\nIteration cap ({MAX_ITERATIONS}) reached without meeting threshold {ROI_ACCEPT_THRESHOLD}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument("--predict-months", type=int, default=DEFAULT_PREDICT_MONTHS)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    args = parser.parse_args()

    run_loop(args.ticker, args.train_months, args.predict_months, args.gap_days)


if __name__ == "__main__":
    main()
