import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

from agents.model_designer import CONTRACT_VERSION, generate_model_script
from wfo.run_iteration import ITERATIONS_DIR, run_iteration
from wfo.schema import IterationResults, load_results

MAX_ITERATIONS = 10
CUMULATIVE_ROI_ACCEPT_THRESHOLD = 4.00
DEFAULT_TRAIN_MONTHS = 3
DEFAULT_PREDICT_MONTHS = 1
DEFAULT_GAP_DAYS = 2
TARGET_MAX_TRADES_PER_MONTH = 100


def chain_roi(period_rois: list[float]) -> float:
    equity = 1.0
    for roi in period_rois:
        equity *= 1 + roi
    return equity - 1


def diagnose(results: IterationResults) -> dict:
    rois = [w.roi for w in results.windows]
    net_rois = [w.net_roi for w in results.windows]
    f1s = [w.f1 for w in results.windows]
    bh_rois = [w.buy_hold_roi for w in results.windows]
    collapsed = sum(1 for w in results.windows if w.f1 == 0.0)
    total_trades = sum(w.num_trades for w in results.windows)
    total_months = len(results.windows) * results.predict_months
    return {
        "avg_roi": statistics.fmean(rois),
        "median_roi": statistics.median(rois),
        "cumulative_roi": chain_roi(rois),
        "cumulative_net_roi": chain_roi(net_rois),
        "avg_f1": statistics.fmean(f1s),
        "avg_buy_hold_roi": statistics.fmean(bh_rois),
        "cumulative_buy_hold_roi": chain_roi(bh_rois),
        "total_trades": total_trades,
        "trades_per_month": total_trades / total_months if total_months else 0.0,
        "num_windows": len(results.windows),
        "collapsed_windows": collapsed,
    }


LONG_ONLY_DIRECTIVE = (
    "This strategy must be LONG-ONLY: never take short positions. Set the module-level "
    "`LONG_ONLY = True` and apply it by clipping the position to `max(position, 0)` before "
    "computing strategy_returns, net_roi, and num_trades, per the contract's long_only support."
)


def build_instructions(
    best_approach: str | None,
    best_diag: dict | None,
    last_approach: str | None,
    last_diag: dict | None,
) -> str:
    if best_diag is None:
        return (
            f"{LONG_ONLY_DIRECTIVE} Design an initial model for intraday minute-bar signal "
            "prediction under this long-only constraint. Pick your own feature set, model, and "
            "prediction framing."
        )

    parts = [LONG_ONLY_DIRECTIVE]
    parts.append(
        f"BEST RESULT SO FAR ({best_approach}): cumulative NET ROI={best_diag['cumulative_net_roi']:+.3f} "
        f"(gross {best_diag['cumulative_roi']:+.3f} vs buy-and-hold {best_diag['cumulative_buy_hold_roi']:+.3f}), "
        f"avg F1={best_diag['avg_f1']:.3f}, at {best_diag['trades_per_month']:.0f} trades/month. Your job is "
        "to beat this specifically — build on what's working rather than abandoning it for something "
        "unrelated unless you have a specific reason to believe it will do better."
    )

    is_regression = last_diag is not best_diag and last_diag["cumulative_net_roi"] < best_diag["cumulative_net_roi"]
    if is_regression:
        parts.append(
            f"Your most recent attempt ({last_approach}) REGRESSED from the best: net ROI "
            f"{last_diag['cumulative_net_roi']:+.3f} at {last_diag['trades_per_month']:.0f} trades/month. "
            "Don't continue further in that direction — return toward the best result's approach and "
            "trade frequency, and refine from there instead of drifting further away from it."
        )

    if last_diag["cumulative_roi"] <= last_diag["cumulative_buy_hold_roi"]:
        parts.append(
            "The most recent attempt did not beat simply buying and holding the stock over the same "
            "periods — that's the real bar to clear, not just positive ROI."
        )
    if last_diag["cumulative_net_roi"] < last_diag["cumulative_roi"] - 0.05:
        parts.append(
            f"Note: the most recent attempt's net-of-fees cumulative ROI "
            f"({last_diag['cumulative_net_roi']:+.3f}) is meaningfully worse than gross "
            f"({last_diag['cumulative_roi']:+.3f}) — it took {last_diag['trades_per_month']:.0f} "
            f"trades/month. At a flat $1/trade fee, anything much above {TARGET_MAX_TRADES_PER_MONTH} "
            "trades/month tends to get dominated by fees almost regardless of edge. Don't just trade "
            f"'a bit less' — substantially lengthen the horizon (e.g. 4-8x longer) and/or raise the "
            f"confidence/magnitude threshold so the flat/no-trade case fires much more often, aiming "
            f"for roughly {TARGET_MAX_TRADES_PER_MONTH // 2}-{TARGET_MAX_TRADES_PER_MONTH} "
            "trades/month, not several hundred."
        )
    if last_diag["collapsed_windows"] > last_diag["num_windows"] / 2:
        parts.append(
            "More than half of the most recent attempt's windows collapsed to a constant prediction — "
            "address class imbalance or add stronger discriminative features rather than just retuning."
        )
    parts.append(
        "Improve on the BEST result above: aim for higher and more consistent NET ROI across windows "
        "(not just gross), a higher CUMULATIVE (compounded) return across the whole walk-forward "
        "stretch, and a trade frequency close to what's already working unless you have good reason "
        "to move it."
    )
    return " ".join(parts)


def print_iteration_report(iteration_num: int, results: IterationResults, diag: dict, prev_diag: dict | None):
    print(f"\n[iteration {iteration_num}/{MAX_ITERATIONS}]  ticker: {results.ticker}")
    print(f"  approach: {results.approach}")
    print(f"  prediction target: {results.prediction_target}")
    if prev_diag is not None:
        print(f"  cumulative ROI:     {prev_diag['cumulative_roi']:+.3f} -> {diag['cumulative_roi']:+.3f}  (cumulative buy&hold: {diag['cumulative_buy_hold_roi']:+.3f})")
        print(f"  cumulative net ROI: {prev_diag['cumulative_net_roi']:+.3f} -> {diag['cumulative_net_roi']:+.3f}  ($1/trade fee, $10k capital)")
        print(f"  avg ROI:            {prev_diag['avg_roi']:+.3f} -> {diag['avg_roi']:+.3f}  (avg buy&hold: {diag['avg_buy_hold_roi']:+.3f})")
        print(f"  avg F1:             {prev_diag['avg_f1']:.3f} -> {diag['avg_f1']:.3f}")
    else:
        print(f"  cumulative ROI:     {diag['cumulative_roi']:+.3f}  (cumulative buy&hold: {diag['cumulative_buy_hold_roi']:+.3f})")
        print(f"  cumulative net ROI: {diag['cumulative_net_roi']:+.3f}  ($1/trade fee, $10k capital)")
        print(f"  avg ROI:            {diag['avg_roi']:+.3f}  (avg buy&hold: {diag['avg_buy_hold_roi']:+.3f})")
        print(f"  avg F1:             {diag['avg_f1']:.3f}")
    print(f"  windows: {diag['num_windows']} ({diag['collapsed_windows']} collapsed to constant prediction)")
    print(f"  total trades: {diag['total_trades']}  ({diag['trades_per_month']:.0f}/month)")
    for w in results.windows:
        print(
            f"    window {w.index:2d}  f1={w.f1:.3f}  acc={w.accuracy:.3f}  trades={w.num_trades:4d}  "
            f"roi={w.roi:+.3f}  net_roi={w.net_roi:+.3f}  buy&hold={w.buy_hold_roi:+.3f}  maxdd={w.max_drawdown:+.3f}"
        )


def run_loop(
    tickers: list[str],
    train_months: int,
    predict_months: int,
    gap_days: int,
    holdout_months: int = 0,
    holdout_tickers: list[str] | None = None,
) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    holdout_tickers = holdout_tickers or []
    last_approach, last_diag = None, None
    best_approach, best_diag = None, None
    best_script_path = None

    print(f"Rotating search tickers: {', '.join(tickers)}")
    if holdout_months:
        print(f"Also reserving the trailing {holdout_months} month(s) of each search ticker as a "
              "time-based blind holdout.")
    if holdout_tickers:
        print(f"Reserving entire tickers never used in search: {', '.join(holdout_tickers)} "
              "(full history available for a large-sample blind test afterward).")

    for i in range(1, MAX_ITERATIONS + 1):
        ticker = tickers[(i - 1) % len(tickers)]
        instructions = build_instructions(best_approach, best_diag, last_approach, last_diag)
        tag = f"{run_id}_{ticker}_iter{i}"

        try:
            script_path = generate_model_script(instructions, iteration_tag=tag)
        except Exception as e:
            print(f"\n[iteration {i}/{MAX_ITERATIONS}] ticker: {ticker}  FAILED: generation error: {e}")
            continue

        ok, outcome = run_iteration(
            script_path, ticker, train_months, predict_months, gap_days,
            iteration_id=tag, contract_version=CONTRACT_VERSION, holdout_months=holdout_months,
        )
        if not ok:
            print(f"\n[iteration {i}/{MAX_ITERATIONS}] ticker: {ticker}  FAILED: {outcome}")
            continue

        results = load_results(ITERATIONS_DIR / outcome / "results.json")
        new_diag = diagnose(results)
        print_iteration_report(i, results, new_diag, last_diag)

        last_approach, last_diag = results.approach, new_diag
        if best_diag is None or new_diag["cumulative_net_roi"] > best_diag["cumulative_net_roi"]:
            best_approach, best_diag = last_approach, last_diag
            best_script_path = script_path
            print(f"  ** new best (net ROI {best_diag['cumulative_net_roi']:+.3f}) **")

        if new_diag["cumulative_net_roi"] >= CUMULATIVE_ROI_ACCEPT_THRESHOLD:
            print(
                f"\nACCEPTED at iteration {i}: cumulative NET ROI {new_diag['cumulative_net_roi']:.3f} "
                f">= threshold {CUMULATIVE_ROI_ACCEPT_THRESHOLD} "
                f"(gross was {new_diag['cumulative_roi']:.3f})"
            )
            _print_holdout_hint(
                holdout_months, holdout_tickers, best_script_path, train_months, predict_months, gap_days
            )
            return

    _print_holdout_hint(
        holdout_months, holdout_tickers, best_script_path, train_months, predict_months, gap_days
    )
    print(
        f"\nIteration cap ({MAX_ITERATIONS}) reached without meeting cumulative NET ROI threshold "
        f"{CUMULATIVE_ROI_ACCEPT_THRESHOLD}. Best found: net ROI {best_diag['cumulative_net_roi']:+.3f} "
        f"({best_approach})"
    )


def _print_holdout_hint(holdout_months, holdout_tickers, best_script_path, train_months, predict_months, gap_days):
    if best_script_path is None:
        return
    if holdout_months:
        print(
            f"\nTo blind-test the best model against its reserved {holdout_months}-month time holdout, "
            f"rerun it with --holdout-months 0 and inspect the trailing windows:\n"
            f"  PYTHONPATH=src python -m wfo.run_iteration {best_script_path} --ticker <search ticker> "
            f"--train-months {train_months} --predict-months {predict_months} --gap-days {gap_days} "
            f"--holdout-months 0"
        )
    for ht in holdout_tickers:
        print(
            f"\nTo blind-test the best model against the fully-reserved ticker {ht} (full history, "
            f"never seen during search):\n"
            f"  PYTHONPATH=src python -m wfo.run_iteration {best_script_path} --ticker {ht} "
            f"--train-months {train_months} --predict-months {predict_months} --gap-days {gap_days} "
            f"--holdout-months 0"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers to rotate through during search, e.g. AAPL,NVDA,AMZN")
    parser.add_argument("--holdout-tickers", default="", help="Comma-separated tickers to reserve entirely (never used in search), e.g. MSFT,SPY")
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument("--predict-months", type=int, default=DEFAULT_PREDICT_MONTHS)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    parser.add_argument(
        "--holdout-months", type=int, default=0,
        help="Additionally reserve this many trailing months from every search ticker's WFO windows.",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    holdout_tickers = [t.strip() for t in args.holdout_tickers.split(",") if t.strip()]

    run_loop(
        tickers, args.train_months, args.predict_months, args.gap_days,
        args.holdout_months, holdout_tickers,
    )


if __name__ == "__main__":
    main()
