"""Precompute the replay demo into a static, dependency-free page.

Gradio Spaces need a paid plan; static Spaces are free. Every number the page
shows is computed here by ReplayLab, at build time, over the same recorded
workload the Gradio app uses. The page itself only looks values up.

The one thing it computes in the browser is which cases changed between two
thresholds, and that is not replay math: `_escalated` is `p >= threshold`, so
escalation is monotone in the threshold and the changed set is exactly the
events whose probability lies between the two. That is a range count over a
sorted list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "spaces" / "demo"))

from lossbench.costs.registry import list_cost_profiles, load_cost_profile  # noqa: E402
from lossbench.replay.simulator import ReplayLab  # noqa: E402
from lossbench.schema import PolicyBundle  # noqa: E402

GRID_STEPS = 101  # thresholds 0.00 .. 1.00 inclusive, step 0.01
WORKLOAD_SEED = 7
WORKLOAD_TASKS = 200


def _thresholds() -> list[float]:
    return [round(i / (GRID_STEPS - 1), 2) for i in range(GRID_STEPS)]


def build_data() -> dict:
    """Run the workload once and score every threshold with ReplayLab."""
    import demo  # from spaces/demo, added to sys.path above

    events, _ = demo.build_workload(WORKLOAD_SEED, WORKLOAD_TASKS)
    thresholds = _thresholds()

    curves: dict[str, dict] = {}
    for name in list_cost_profiles():
        lab = ReplayLab(load_cost_profile(name))
        losses, loads = [], []
        for tau in thresholds:
            # Same threshold both sides: before == after == the value at tau.
            outcome = lab.simulate(
                events, PolicyBundle(id="s", cost_model_id=name, escalation_threshold=tau), tau
            )
            losses.append(outcome.before_loss)
            loads.append(outcome.before_review_load)
        curves[name] = {"loss": losses, "review_load": loads}

    rows = [
        {
            "id": event.event_id[:8],
            "p": event.calibrated_probability,
            "decision": event.decision.value,
        }
        for event in events
        if event.calibrated_probability is not None
    ]
    rows.sort(key=lambda r: r["p"])

    return {
        "thresholds": thresholds,
        "curves": curves,
        "events": rows,
        "n_events": len(events),
        "seed": WORKLOAD_SEED,
        "n_tasks": WORKLOAD_TASKS,
        "default_cost_model": "reconciliation",
    }


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LossBench control plane</title>
<style>
  :root {
    --bg: #ffffff; --fg: #14181f; --muted: #5a6472; --line: #e2e6ec;
    --card: #f7f8fa; --accent: #1f6feb; --warn: #9a3412;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1319; --fg: #e6e9ee; --muted: #97a1b0; --line: #262d38;
      --card: #161b23; --accent: #6aa8ff; --warn: #f0a675;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
    font: 16px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 60rem; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin: 0 0 .35rem; letter-spacing: -.02em; }
  .sub { color: var(--muted); margin: 0 0 1.75rem; }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;
  }
  .controls { display: grid; gap: 1.1rem; grid-template-columns: 1fr; }
  @media (min-width: 46rem) { .controls { grid-template-columns: 1fr 1fr; } }
  label { display: block; font-weight: 600; font-size: .85rem; margin-bottom: .4rem; }
  label span { float: right; font-variant-numeric: tabular-nums; color: var(--accent); }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  select {
    width: 100%; padding: .5rem .6rem; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
  }
  .stats { display: grid; gap: 1rem; grid-template-columns: repeat(2, 1fr); }
  @media (min-width: 46rem) { .stats { grid-template-columns: repeat(4, 1fr); } }
  .stat { border-left: 3px solid var(--line); padding-left: .8rem; }
  .stat .k {
    font-size: .75rem; text-transform: uppercase;
    letter-spacing: .06em; color: var(--muted);
  }
  .stat .v { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
  .stat .d { font-size: .8rem; color: var(--muted); font-variant-numeric: tabular-nums; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td {
    text-align: left; padding: .4rem .7rem;
    border-bottom: 1px solid var(--line); white-space: nowrap;
  }
  th { color: var(--muted); font-weight: 600; }
  td.num { font-variant-numeric: tabular-nums; }
  .note { font-size: .85rem; color: var(--muted); }
  .note strong { color: var(--warn); font-weight: 600; }
  code { background: var(--card); padding: .1rem .3rem; border-radius: 4px; font-size: .9em; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Re-run last month under a different risk policy</h1>
  <p class="sub">Replay a recorded reconciliation workload at a new escalation
  threshold and read the counterfactual. No model calls, fully deterministic.</p>

  <div class="panel">
    <div class="controls">
      <div>
        <label for="before">Last month's threshold <span id="beforeV">0.50</span></label>
        <input type="range" id="before" min="0" max="1" step="0.01" value="0.5">
      </div>
      <div>
        <label for="after">New threshold <span id="afterV">0.70</span></label>
        <input type="range" id="after" min="0" max="1" step="0.01" value="0.7">
      </div>
      <div>
        <label for="cm">Cost model</label>
        <select id="cm"></select>
      </div>
      <div>
        <label for="none">Workload</label>
        <div class="note" id="meta"></div>
      </div>
    </div>
  </div>

  <div class="panel stats">
    <div class="stat"><div class="k">Loss before</div><div class="v" id="lb">-</div></div>
    <div class="stat"><div class="k">Loss after</div><div class="v" id="la">-</div>
      <div class="d" id="ld"></div></div>
    <div class="stat"><div class="k">Review load before</div><div class="v" id="rb">-</div></div>
    <div class="stat"><div class="k">Review load after</div><div class="v" id="ra">-</div>
      <div class="d" id="rd"></div></div>
  </div>

  <div class="panel">
    <p class="note" id="changed"></p>
    <div class="scroll"><table>
      <thead><tr><th>event</th><th>before</th><th>after</th><th>confidence</th></tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
  </div>

  <p class="note"><strong>Read this before quoting a number.</strong>
  Loss is severity-weighted: an unreviewed error is charged the cost
  <code>K</code> of the task it got wrong, and every escalated case is charged
  the review price. Severity costs are contested inputs, which is why the cost
  model is a control rather than a constant. This is a recorded
  <em>stub</em> workload, so confidences are derived from task difficulty
  rather than measured from a model, and the stub never errs &mdash; with a
  perfect model, escalation is pure cost. Real model results live in the
  dataset, not here.</p>

  <p class="note">Every loss and review-load figure on this page was computed
  by <code>ReplayLab</code> at build time over all 101 thresholds. The page
  looks them up. <a href="https://github.com/caiotheodoro/lossbench">Source</a>
  &middot; <a href="https://huggingface.co/datasets/caiotheodoro/lossbench-finance-v1">Dataset</a></p>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("payload").textContent);
  const $ = (id) => document.getElementById(id);
  const idx = (t) => Math.round(t * (D.thresholds.length - 1));
  const ps = D.events.map((e) => e.p); // already sorted ascending

  const cm = $("cm");
  Object.keys(D.curves).forEach((name) => {
    const o = document.createElement("option");
    o.value = o.textContent = name;
    if (name === D.default_cost_model) o.selected = true;
    cm.appendChild(o);
  });
  $("meta").textContent =
    D.n_events + " recorded events, seed " + D.seed + ", " + D.n_tasks + " tasks";

  // Escalation is p >= threshold, so the cases that change between two
  // thresholds are exactly those whose confidence lies in [lo, hi).
  const lowerBound = (arr, x) => {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
    return lo;
  };

  function fmt(n) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: 4 }); }
  function delta(a, b) {
    const d = b - a;
    if (Math.abs(d) < 1e-9) return "no change";
    return (d > 0 ? "+" : "") + fmt(d);
  }

  function render() {
    const a = parseFloat($("before").value), b = parseFloat($("after").value);
    $("beforeV").textContent = a.toFixed(2);
    $("afterV").textContent = b.toFixed(2);
    const c = D.curves[cm.value];

    const lb = c.loss[idx(a)], la = c.loss[idx(b)];
    const rb = c.review_load[idx(a)], ra = c.review_load[idx(b)];
    $("lb").textContent = fmt(lb);
    $("la").textContent = fmt(la);
    $("ld").textContent = delta(lb, la);
    $("rb").textContent = (rb * 100).toFixed(1) + "%";
    $("ra").textContent = (ra * 100).toFixed(1) + "%";
    $("rd").textContent = delta(rb, ra);

    const lo = Math.min(a, b), hi = Math.max(a, b);
    const from = lowerBound(ps, lo), to = lowerBound(ps, hi);
    const changed = D.events.slice(from, to);
    $("changed").textContent = changed.length === 0
      ? "No case changes decision between these two thresholds."
      : changed.length + " of " + D.n_events + " cases change decision.";

    const body = $("rows");
    body.innerHTML = "";
    changed.slice(0, 200).forEach((e) => {
      const escBefore = e.p >= a, escAfter = e.p >= b;
      const tr = document.createElement("tr");
      [e.id,
       escBefore ? "ESCALATE" : e.decision,
       escAfter ? "ESCALATE" : e.decision,
       e.p.toFixed(3)
      ].forEach((v, i) => {
        const td = document.createElement("td");
        td.textContent = v;
        if (i === 3) td.className = "num";
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  ["before", "after"].forEach((id) => $(id).addEventListener("input", render));
  cm.addEventListener("change", render);
  render();
})();
</script>
</body>
</html>
"""


def build_page(data: dict) -> str:
    """Inline the precomputed payload into the page."""
    blob = json.dumps(data, separators=(",", ":"))
    if "</script" in blob:
        raise ValueError("payload would close the script tag")
    return _PAGE.replace("__DATA__", blob)


__all__ = ["build_data", "build_page"]
