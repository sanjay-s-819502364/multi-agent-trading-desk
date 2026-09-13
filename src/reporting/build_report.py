import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd

ITERATIONS_DIR = Path("iterations")
DATA_DIR = Path("data/minute_aggs")
REPORT_PATH = Path("reports/index.html")
RUN_TIMESTAMP_RE = re.compile(r"^(\d{8}T\d{6})")


def parse_run_timestamp(run_name: str) -> str | None:
    match = RUN_TIMESTAMP_RE.match(run_name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_iteration(dir_path: Path) -> dict | None:
    results_path = dir_path / "results.json"
    if not results_path.exists():
        return None
    payload = json.loads(results_path.read_text())
    windows = payload.get("windows", [])
    if not windows:
        return None

    rois = [w["roi"] for w in windows]
    net_rois = [w["net_roi"] for w in windows if "net_roi" in w]
    f1s = [w["f1"] for w in windows]
    accs = [w["accuracy"] for w in windows]
    dds = [w["max_drawdown"] for w in windows]
    bh_rois = [w["buy_hold_roi"] for w in windows if "buy_hold_roi" in w]

    contract_version_path = dir_path / "contract_version.txt"
    contract_version = contract_version_path.read_text().strip() if contract_version_path.exists() else None

    return {
        "run": dir_path.name,
        "generated_at": parse_run_timestamp(dir_path.name),
        "contract_version": contract_version,
        "ticker": payload.get("ticker", "?"),
        "approach": payload.get("approach", "(pre-dates approach field)"),
        "prediction_target": payload.get("prediction_target", "?"),
        "train_months": payload.get("train_months"),
        "predict_months": payload.get("predict_months"),
        "gap_days": payload.get("gap_days"),
        "cumulative_roi": chain_roi(rois),
        "cumulative_net_roi": chain_roi(net_rois) if net_rois else None,
        "cumulative_buy_hold_roi": chain_roi(bh_rois) if bh_rois else None,
        "avg_roi": statistics.fmean(rois),
        "avg_buy_hold_roi": statistics.fmean(bh_rois) if bh_rois else None,
        "avg_f1": statistics.fmean(f1s),
        "avg_accuracy": statistics.fmean(accs),
        "avg_max_drawdown": statistics.fmean(dds),
        "total_trades": sum(w["num_trades"] for w in windows) if all("num_trades" in w for w in windows) else None,
        "num_windows": len(windows),
        "collapsed_windows": sum(1 for w in windows if w["f1"] == 0.0),
        "windows": windows,
    }


def chain_roi(period_rois: list[float]) -> float:
    equity = 1.0
    for roi in period_rois:
        equity *= 1 + roi
    return equity - 1


def true_full_period_benchmark(ticker: str) -> dict | None:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) < 2:
        return None
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("dt")
    return {
        "ticker": ticker,
        "roi": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1),
        "start": df["dt"].iloc[0].strftime("%Y-%m-%d"),
        "end": df["dt"].iloc[-1].strftime("%Y-%m-%d"),
    }


def collect_benchmarks(rows: list[dict]) -> list[dict]:
    tickers = sorted({row["ticker"] for row in rows})
    benchmarks = [true_full_period_benchmark(t) for t in tickers]
    return [b for b in benchmarks if b is not None]


def collect_iterations() -> list[dict]:
    if not ITERATIONS_DIR.exists():
        return []
    rows = []
    for dir_path in sorted(ITERATIONS_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        row = load_iteration(dir_path)
        if row is not None:
            rows.append(row)
    return rows


def render_html(rows: list[dict], benchmarks: list[dict]) -> str:
    data_json = json.dumps(rows)
    benchmarks_json = json.dumps(benchmarks)
    return HTML_TEMPLATE.replace("__DATA__", data_json).replace("__BENCHMARKS__", benchmarks_json)


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Model Iteration Report</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 24px; background: #f7f7f8; color: #1a1a1a;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #16171a; color: #e8e8ea; }
    th { background: #232428 !important; }
    tr:hover { background: #1f2023 !important; }
    .card { background: #1f2023 !important; border-color: #2c2d31 !important; }
    .approach { color: #b7b9c0 !important; }
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .subtitle { color: #777; font-size: 13px; margin-bottom: 20px; }
  .card { background: #fff; border: 1px solid #e2e2e4; border-radius: 10px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 10px 12px; white-space: nowrap; }
  th {
    background: #eee; cursor: pointer; user-select: none; position: sticky; top: 0;
    font-weight: 600;
  }
  th:hover { background: #e0e0e0; }
  th.sorted-asc::after { content: " \\25B2"; }
  th.sorted-desc::after { content: " \\25BC"; }
  tbody tr { border-top: 1px solid #eee; cursor: pointer; }
  tbody tr:hover { background: #fafafa; }
  .approach { max-width: 420px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #444; font-size: 12px; }
  .pos { color: #1a7f37; font-weight: 600; }
  .neg { color: #c0392b; font-weight: 600; }
  .badge {
    display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; font-weight: 600;
  }
  .badge-warn { background: #fdecea; color: #c0392b; }
  .badge-ok { background: #e6f4ea; color: #1a7f37; }
  .detail-row td { background: #fbfbfc; padding: 16px; }
  .detail-table { width: 100%; font-size: 12px; border-collapse: collapse; }
  .detail-table th, .detail-table td { padding: 4px 8px; }
  .detail-table th { background: transparent; cursor: default; position: static; }
  [hidden] { display: none !important; }
  .benchmarks {
    font-size: 12px; color: #666; margin-bottom: 16px; padding: 10px 14px;
    background: #f0f0f2; border-radius: 8px; border: 1px solid #e2e2e4;
  }
  @media (prefers-color-scheme: dark) {
    .benchmarks { background: #1f2023 !important; border-color: #2c2d31 !important; color: #a8a9ae !important; }
  }
  .benchmarks b { color: inherit; }
</style>
</head>
<body>
<h1>Model Iteration Report</h1>
<div class="subtitle" id="subtitle"></div>
<div class="benchmarks" id="benchmarks"></div>
<div class="card">
  <table id="report-table">
    <thead>
      <tr>
        <th data-key="run">Run</th>
        <th data-key="generated_at">Generated</th>
        <th data-key="ticker">Ticker</th>
        <th data-key="contract_version">Contract</th>
        <th data-key="approach">Approach</th>
        <th data-key="prediction_target">Target</th>
        <th data-key="cumulative_roi">Cum. ROI</th>
        <th data-key="cumulative_net_roi">Cum. Net ROI</th>
        <th data-key="cumulative_buy_hold_roi">Cum. Buy&amp;Hold</th>
        <th data-key="avg_roi">Avg ROI</th>
        <th data-key="avg_buy_hold_roi">Avg Buy&amp;Hold</th>
        <th data-key="avg_f1">Avg F1</th>
        <th data-key="avg_accuracy">Avg Acc</th>
        <th data-key="avg_max_drawdown">Avg MaxDD</th>
        <th data-key="num_windows">Windows</th>
        <th data-key="total_trades">Trades</th>
        <th data-key="collapsed_windows">Collapsed</th>
      </tr>
    </thead>
    <tbody id="report-body"></tbody>
  </table>
</div>

<script>
const DATA = __DATA__;
const BENCHMARKS = __BENCHMARKS__;
let sortKey = "generated_at";
let sortDir = -1;

function renderBenchmarks() {
  const el = document.getElementById("benchmarks");
  if (!BENCHMARKS.length) { el.hidden = true; return; }
  const parts = BENCHMARKS.map(b =>
    `<b>${b.ticker}</b>: continuous buy&amp;hold ${b.start} → ${b.end} = <span class="${pctClass(b.roi)}">${fmtPct(b.roi)}</span>`
  );
  el.innerHTML = "Reference only (full data range, continuous hold) &mdash; " +
    "NOT directly comparable to the Cum. columns below, which only cover the walk-forward " +
    "windows actually tested and lose the initial train/gap period plus overnight gaps at window seams:<br>" +
    parts.join(" &nbsp;|&nbsp; ");
}

function pctClass(v) { return v >= 0 ? "pos" : "neg"; }
function fmtPct(v) { return (v * 100).toFixed(1) + "%"; }

function renderDetail(row) {
  let html = '<table class="detail-table"><thead><tr>' +
    '<th>#</th><th>Train</th><th>Predict</th><th>F1</th><th>Acc</th><th>Trades</th><th>ROI</th><th>Net ROI</th><th>Buy&amp;Hold</th><th>MaxDD</th>' +
    '</tr></thead><tbody>';
  for (const w of row.windows) {
    const bh = "buy_hold_roi" in w
      ? `<td class="${pctClass(w.buy_hold_roi)}">${fmtPct(w.buy_hold_roi)}</td>`
      : '<td>—</td>';
    const net = "net_roi" in w
      ? `<td class="${pctClass(w.net_roi)}">${fmtPct(w.net_roi)}</td>`
      : '<td>—</td>';
    const trades = "num_trades" in w ? w.num_trades : "—";
    html += `<tr>
      <td>${w.index}</td>
      <td>${w.train_start} to ${w.train_end}</td>
      <td>${w.predict_start} to ${w.predict_end}</td>
      <td>${w.f1.toFixed(3)}</td>
      <td>${w.accuracy.toFixed(3)}</td>
      <td>${trades}</td>
      <td class="${pctClass(w.roi)}">${fmtPct(w.roi)}</td>
      ${net}
      ${bh}
      <td class="neg">${fmtPct(w.max_drawdown)}</td>
    </tr>`;
  }
  html += "</tbody></table>";
  return html;
}

function render() {
  const rows = [...DATA].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av === null) return 1;
    if (bv === null) return -1;
    if (typeof av === "string") return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });

  document.getElementById("subtitle").textContent =
    `${rows.length} iteration${rows.length === 1 ? "" : "s"} found in iterations/`;

  const tbody = document.getElementById("report-body");
  tbody.innerHTML = "";
  for (const row of rows) {
    const collapsedBadge = row.collapsed_windows > row.num_windows / 2
      ? `<span class="badge badge-warn">${row.collapsed_windows}/${row.num_windows}</span>`
      : `<span class="badge badge-ok">${row.collapsed_windows}/${row.num_windows}</span>`;

    const naCell = (v) => v === null
      ? '<span title="pre-dates this field">—</span>'
      : `<span class="${pctClass(v)}">${fmtPct(v)}</span>`;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.run}</td>
      <td>${row.generated_at ?? "—"}</td>
      <td>${row.ticker}</td>
      <td>${row.contract_version ?? "—"}</td>
      <td class="approach" title="${row.approach.replace(/"/g, '&quot;')}">${row.approach}</td>
      <td>${row.prediction_target}</td>
      <td class="${pctClass(row.cumulative_roi)}">${fmtPct(row.cumulative_roi)}</td>
      <td>${naCell(row.cumulative_net_roi)}</td>
      <td>${naCell(row.cumulative_buy_hold_roi)}</td>
      <td class="${pctClass(row.avg_roi)}">${fmtPct(row.avg_roi)}</td>
      <td>${naCell(row.avg_buy_hold_roi)}</td>
      <td>${row.avg_f1.toFixed(3)}</td>
      <td>${row.avg_accuracy.toFixed(3)}</td>
      <td class="neg">${fmtPct(row.avg_max_drawdown)}</td>
      <td>${row.num_windows}</td>
      <td>${row.total_trades ?? "—"}</td>
      <td>${collapsedBadge}</td>
    `;
    const detailTr = document.createElement("tr");
    detailTr.className = "detail-row";
    detailTr.hidden = true;
    const detailTd = document.createElement("td");
    detailTd.colSpan = 17;
    detailTd.innerHTML = renderDetail(row);
    detailTr.appendChild(detailTd);

    tr.addEventListener("click", () => { detailTr.hidden = !detailTr.hidden; });

    tbody.appendChild(tr);
    tbody.appendChild(detailTr);
  }

  for (const th of document.querySelectorAll("th")) {
    th.classList.remove("sorted-asc", "sorted-desc");
    if (th.dataset.key === sortKey) {
      th.classList.add(sortDir === 1 ? "sorted-asc" : "sorted-desc");
    }
  }
}

for (const th of document.querySelectorAll("th[data-key]")) {
  th.addEventListener("click", () => {
    if (sortKey === th.dataset.key) {
      sortDir *= -1;
    } else {
      sortKey = th.dataset.key;
      sortDir = -1;
    }
    render();
  });
}

renderBenchmarks();
render();
</script>
</body>
</html>
"""


def main():
    rows = collect_iterations()
    benchmarks = collect_benchmarks(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_html(rows, benchmarks))
    print(f"Wrote {REPORT_PATH} ({len(rows)} iterations)")


if __name__ == "__main__":
    main()
