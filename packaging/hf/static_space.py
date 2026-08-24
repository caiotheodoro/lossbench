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
from lossbench.replay.simulator import ReplayLab, _severity  # noqa: E402
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
            # task_id, not event_id: every event_id shares an "evt-demo-stub-"
            # prefix, so a truncated one is the same string on every row.
            "id": event.task_id,
            "p": event.calibrated_probability,
            "decision": event.decision.value,
            "severity": _severity(event).value,
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
<title>LossBench — re-run last month</title>
<meta name="description"
 content="Replay a recorded finance workload under a different escalation
 policy and read the counterfactual loss.">
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1/dist/fonts/geist-mono/style.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap">
<style>
  :root {
    --font-sans: "Plus Jakarta Sans", -apple-system, sans-serif;
    --font-mono: "Geist Mono", ui-monospace, monospace;
    --accent: #c93d1e;
    --border: #e8e8e4;
    --border-strong: #d4d4cf;
    --bg: #ffffff;
    --bg-subtle: #f8f8f7;
    --code-bg: #f3f3ed;
    --text: #171717;
    --text-2: #6b6b6b;
    --text-3: #a3a3a3;
    --up: #1f7a3d;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html {
    font-family: var(--font-sans);
    font-size: 16px; line-height: 1.6;
    color: var(--text); background: var(--bg);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  body {
    min-height: 100vh; overflow-x: hidden; position: relative;
    background-color: var(--bg);
    background-image:
      linear-gradient(rgba(0,0,0,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,0,0,.04) 1px, transparent 1px);
    background-size: 36px 36px;
  }
  body::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: linear-gradient(to bottom, transparent 45%, #fff 100%);
  }
  body > * { position: relative; z-index: 1; }
  ::selection { background: var(--accent); color: #fff; }

  main {
    max-width: 960px; margin: 0 auto;
    padding: clamp(2.5rem, 8vh, 5rem) 1.5rem 4rem;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  [data-fade] {
    opacity: 0;
    animation: fadeUp .55s cubic-bezier(.25,.46,.45,.94) forwards;
  }
  [data-delay="1"] { animation-delay: .05s; }
  [data-delay="2"] { animation-delay: .12s; }
  [data-delay="3"] { animation-delay: .19s; }
  [data-delay="4"] { animation-delay: .26s; }
  [data-delay="5"] { animation-delay: .33s; }
  @media (prefers-reduced-motion: reduce) {
    [data-fade] { opacity: 1; animation: none; }
  }

  .kicker {
    font-family: var(--font-mono); font-size: .7rem; letter-spacing: .14em;
    text-transform: uppercase; color: var(--text-3); margin-bottom: 1rem;
  }
  .kicker b { color: var(--accent); font-weight: 500; }

  h1 {
    font-size: clamp(2rem, 5vw, 3rem); font-weight: 800;
    letter-spacing: -.04em; line-height: 1.05; margin-bottom: 1.25rem;
  }
  h1 .dot { color: var(--accent); }

  .standfirst {
    font-size: 1.075rem; line-height: 1.75; color: var(--text-2);
    max-width: 40rem; margin-bottom: 3rem;
  }

  h2 {
    font-size: .72rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: var(--text-3);
    padding-bottom: .6rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  section { margin-bottom: 3.5rem; }

  /* ---------- controls ---------- */
  .controls { display: grid; gap: 2rem; grid-template-columns: 1fr; }
  @media (min-width: 44rem) { .controls { grid-template-columns: 1fr 1fr auto; } }

  .field label {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .82rem; font-weight: 600; margin-bottom: .75rem;
  }
  .field label var {
    font-family: var(--font-mono); font-style: normal;
    font-size: .95rem; color: var(--accent);
  }
  input[type=range] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 1px; background: var(--border-strong);
    outline: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 13px; height: 13px; border-radius: 50%;
    background: var(--bg); border: 1.5px solid var(--accent);
    transition: transform .15s ease;
  }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.25); }
  input[type=range]::-moz-range-thumb {
    width: 13px; height: 13px; border-radius: 50%;
    background: var(--bg); border: 1.5px solid var(--accent);
  }
  input[type=range]:focus-visible { outline: 2px solid var(--accent); outline-offset: 6px; }

  select {
    font-family: var(--font-mono); font-size: .82rem; color: var(--text);
    padding: .45rem 2rem .45rem .7rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6'>\
<path d='M1 1l4 4 4-4' stroke='%23a3a3a3' stroke-width='1.4' fill='none'\
 stroke-linecap='round'/></svg>");
    background-repeat: no-repeat; background-position: right .7rem center;
  }
  select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  /* ---------- curve ---------- */
  .curve-wrap {
    border: 1px solid var(--border); border-radius: 10px;
    padding: 1.5rem 1.25rem 1rem; background: rgba(255,255,255,.6);
  }
  svg.curve { display: block; width: 100%; height: auto; }
  .curve-foot {
    display: flex; justify-content: space-between; gap: 1rem;
    font-family: var(--font-mono); font-size: .68rem; color: var(--text-3);
    margin-top: .5rem;
  }

  /* ---------- ledger ---------- */
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; }
  caption { text-align: left; color: var(--text-2); font-size: .85rem; margin-bottom: .9rem; }
  th {
    font-size: .68rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--text-3);
    text-align: left; padding: 0 1.25rem .6rem 0;
    border-bottom: 1px solid var(--border-strong); white-space: nowrap;
  }
  td {
    padding: .7rem 1.25rem .7rem 0; border-bottom: 1px solid var(--border);
    font-size: .88rem; white-space: nowrap;
  }
  th.r, td.r { text-align: right; padding-right: 0; }
  td.mono { font-family: var(--font-mono); font-size: .8rem; color: var(--text-2); }
  td.big { font-family: var(--font-mono); font-size: 1.35rem; letter-spacing: -.02em; }
  tr:last-child td { border-bottom: none; }
  .row-label { font-weight: 600; }
  .row-label small {
    display: block; font-weight: 400; font-size: .75rem;
    color: var(--text-3); letter-spacing: 0;
  }

  .delta { font-family: var(--font-mono); font-size: .82rem; }
  .delta.down { color: var(--up); }
  .delta.up { color: var(--accent); }
  .delta.flat { color: var(--text-3); }

  .pill {
    display: inline-block; font-family: var(--font-mono); font-size: .68rem;
    letter-spacing: .04em; padding: .15em .5em; border-radius: 4px;
    border: 1px solid var(--border); background: var(--code-bg); color: var(--text-2);
  }
  .pill.esc { border-color: var(--accent); color: var(--accent); background: transparent; }

  td.sev {
    font-family: var(--font-mono); font-size: .7rem; letter-spacing: .06em;
  }
  td.sev-LOW { color: var(--text-3); }
  td.sev-MEDIUM { color: var(--text-2); }
  td.sev-HIGH { color: var(--accent); font-weight: 500; }
  td.sev-CRITICAL { color: var(--accent); font-weight: 700; }

  .empty { color: var(--text-3); font-size: .9rem; padding: 1.5rem 0; }

  /* ---------- notes ---------- */
  .notes { max-width: 44rem; }
  .notes p { font-size: .875rem; line-height: 1.75; color: var(--text-2); margin-bottom: 1rem; }
  .notes p:last-child { margin-bottom: 0; }
  .notes strong { color: var(--text); font-weight: 600; }
  code {
    font-family: var(--font-mono); font-size: .85em; background: var(--code-bg);
    border: 1px solid var(--border); border-radius: 4px; padding: .1em .35em;
  }
  a {
    color: inherit; text-decoration: none;
    border-bottom: 1px solid var(--border-strong);
    transition: border-color .15s, color .15s;
  }
  a:hover { color: var(--accent); border-color: var(--accent); }

  footer {
    margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 1.25rem;
    font-family: var(--font-mono); font-size: .72rem; color: var(--text-3);
  }
</style>
</head>
<body>
<main>
  <p class="kicker" data-fade data-delay="1">LossBench <b>&middot;</b> control plane</p>

  <h1 data-fade data-delay="1">Re-run last month<span class="dot">.</span></h1>

  <p class="standfirst" data-fade data-delay="2">
    Every decision was recorded with the risk the model reported at the time.
    Move the escalation threshold and the whole month is re-decided
    &mdash; no model calls, nothing re-run, fully deterministic.
  </p>

  <section data-fade data-delay="2">
    <h2>Policy</h2>
    <div class="controls">
      <div class="field">
        <label for="before">Last month <var id="beforeV">0.50</var></label>
        <input type="range" id="before" min="0" max="1" step="0.01" value="0.5">
      </div>
      <div class="field">
        <label for="after">Proposed <var id="afterV">0.70</var></label>
        <input type="range" id="after" min="0" max="1" step="0.01" value="0.7">
      </div>
      <div class="field">
        <label for="cm">Cost model</label>
        <select id="cm"></select>
      </div>
    </div>
  </section>

  <section data-fade data-delay="3">
    <h2>Expected loss across every threshold</h2>
    <div class="curve-wrap">
      <svg class="curve" viewBox="0 0 720 200" preserveAspectRatio="none"
           role="img" aria-labelledby="curveTitle">
        <title id="curveTitle">Expected loss as a function of the escalation threshold</title>
        <g id="curveG"></g>
      </svg>
      <div class="curve-foot">
        <span>0.00 &mdash; escalate everything</span>
        <span id="curveMax"></span>
        <span>1.00 &mdash; escalate nothing</span>
      </div>
    </div>
  </section>

  <section data-fade data-delay="4">
    <h2>Counterfactual</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Measure</th>
            <th class="r">Last month</th>
            <th class="r">Proposed</th>
            <th class="r">Change</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="row-label">Expected loss<small>severity-weighted</small></td>
            <td class="r big" id="lb">&mdash;</td>
            <td class="r big" id="la">&mdash;</td>
            <td class="r"><span class="delta" id="ld"></span></td>
          </tr>
          <tr>
            <td class="row-label">Review load<small>share sent to a human</small></td>
            <td class="r big" id="rb">&mdash;</td>
            <td class="r big" id="ra">&mdash;</td>
            <td class="r"><span class="delta" id="rd"></span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <section data-fade data-delay="5">
    <h2>Cases that change hands</h2>
    <div class="scroll">
      <table>
        <caption id="changed"></caption>
        <thead>
          <tr>
            <th>Task</th>
            <th>Severity</th>
            <th>Last month</th>
            <th>Proposed</th>
            <th class="r">Reported risk</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <p class="empty" id="empty" hidden>No case changes hands between these two thresholds.</p>
  </section>

  <section class="notes" data-fade data-delay="5">
    <h2>Read this before quoting a number</h2>
    <p>
      Loss is severity-weighted. An unreviewed error is charged the cost
      <code>K</code> of the task it got wrong, and every escalated case is
      charged the review price, so one HIGH miss outweighs a pile of LOW ones.
      Severity costs are contested inputs rather than constants, which is why
      the cost model is a control on this page and not a hard-coded number.
    </p>
    <p>
      <strong>This is a recorded stub workload.</strong> Reported risk is
      derived from task difficulty rather than measured from a model, and the
      stub never gets an answer wrong &mdash; against a perfect model,
      escalation is pure cost and the curve only falls. Real model results
      belong in the dataset, not here.
    </p>
    <p>
      Every loss and review-load figure was computed by <code>ReplayLab</code>
      at build time across all 101 thresholds and four cost models. The page
      looks them up. The one thing computed in your browser is which cases
      change hands, and that is a range count, not replay maths: escalation is
      <code>risk &gt;= threshold</code>, so the cases that move are exactly
      those whose risk falls between the two.
    </p>
  </section>

  <footer>
    <a href="https://github.com/caiotheodoro/lossbench">Source</a>
    <a href="https://huggingface.co/datasets/caiotheodoro/lossbench-finance-v1">Dataset</a>
    <span id="meta"></span>
  </footer>
</main>

<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  var D = JSON.parse(document.getElementById("payload").textContent);
  var $ = function (id) { return document.getElementById(id); };
  var STEPS = D.thresholds.length - 1;
  var idx = function (t) { return Math.round(t * STEPS); };
  var risks = D.events.map(function (e) { return e.p; }); // sorted ascending
  var NS = "http://www.w3.org/2000/svg";
  var W = 720, H = 200, PAD = 6;

  var cm = $("cm");
  Object.keys(D.curves).forEach(function (name) {
    var o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    if (name === D.default_cost_model) o.selected = true;
    cm.appendChild(o);
  });
  $("meta").textContent = D.n_events + " events · seed " + D.seed + " · " + D.n_tasks + " tasks";

  function el(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  // Escalation is risk >= threshold, so the cases that move between two
  // thresholds are exactly those whose risk lies in [lo, hi).
  function lowerBound(arr, x) {
    var lo = 0, hi = arr.length;
    while (lo < hi) { var m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
    return lo;
  }
  function num(n) {
    return Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  function pct(n) { return (n * 100).toFixed(1) + "%"; }

  function setDelta(node, before, after, fmt) {
    var d = after - before;
    if (Math.abs(d) < 1e-9) {
      node.textContent = "no change"; node.className = "delta flat"; return;
    }
    var pctChange = before > 0 ? " (" + (d / before * 100).toFixed(0) + "%)" : "";
    node.textContent = (d > 0 ? "+" : "\\u2212") + fmt(Math.abs(d)) + pctChange;
    node.className = "delta " + (d > 0 ? "up" : "down");
  }

  function drawCurve(curve, a, b) {
    var g = $("curveG");
    while (g.firstChild) g.removeChild(g.firstChild);
    var max = Math.max.apply(null, curve.loss) || 1;
    $("curveMax").textContent = "peak " + num(max);

    var x = function (i) { return PAD + (i / STEPS) * (W - PAD * 2); };
    var y = function (v) { return H - PAD - (v / max) * (H - PAD * 2); };

    // band between the two thresholds
    var xa = x(idx(a)), xb = x(idx(b));
    g.appendChild(el("rect", {
      x: Math.min(xa, xb), y: PAD, width: Math.abs(xb - xa) || 1, height: H - PAD * 2,
      fill: "#c93d1e", "fill-opacity": ".06"
    }));

    var pts = curve.loss.map(function (v, i) { return x(i) + "," + y(v); }).join(" ");
    g.appendChild(el("polygon", {
      points: PAD + "," + (H - PAD) + " " + pts + " " + (W - PAD) + "," + (H - PAD),
      fill: "#c93d1e", "fill-opacity": ".05"
    }));
    g.appendChild(el("polyline", {
      points: pts, fill: "none", stroke: "#171717",
      "stroke-width": "1.5", "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke"
    }));

    [[xa, a, curve.loss[idx(a)], "#a3a3a3"], [xb, b, curve.loss[idx(b)], "#c93d1e"]]
      .forEach(function (m) {
        g.appendChild(el("line", {
          x1: m[0], y1: PAD, x2: m[0], y2: H - PAD,
          stroke: m[3], "stroke-width": "1", "stroke-dasharray": "3 3",
          "vector-effect": "non-scaling-stroke"
        }));
        g.appendChild(el("circle", {
          cx: m[0], cy: y(m[2]), r: "4", fill: "#fff",
          stroke: m[3], "stroke-width": "1.75", "vector-effect": "non-scaling-stroke"
        }));
      });
  }

  function render() {
    var a = parseFloat($("before").value), b = parseFloat($("after").value);
    $("beforeV").textContent = a.toFixed(2);
    $("afterV").textContent = b.toFixed(2);
    var c = D.curves[cm.value];

    var lb = c.loss[idx(a)], la = c.loss[idx(b)];
    var rb = c.review_load[idx(a)], ra = c.review_load[idx(b)];
    $("lb").textContent = num(lb);
    $("la").textContent = num(la);
    $("rb").textContent = pct(rb);
    $("ra").textContent = pct(ra);
    setDelta($("ld"), lb, la, num);
    setDelta($("rd"), rb, ra, pct);

    drawCurve(c, a, b);

    var lo = Math.min(a, b), hi = Math.max(a, b);
    var moved = D.events.slice(lowerBound(risks, lo), lowerBound(risks, hi));
    $("changed").textContent = moved.length
      ? moved.length + " of " + D.n_events + " cases change hands."
      : "";
    $("empty").hidden = moved.length > 0;

    var body = $("rows");
    body.innerHTML = "";
    moved.slice(0, 150).forEach(function (e) {
      var tr = document.createElement("tr");
      var cells = [
        { t: e.id, c: "mono" },
        { t: e.severity, c: "sev sev-" + e.severity },
        { t: e.p >= a ? "ESCALATE" : e.decision, pill: e.p >= a },
        { t: e.p >= b ? "ESCALATE" : e.decision, pill: e.p >= b },
        { t: e.p.toFixed(3), c: "mono r" }
      ];
      cells.forEach(function (cell) {
        var td = document.createElement("td");
        if (cell.c) td.className = cell.c;
        if (cell.pill !== undefined) {
          var span = document.createElement("span");
          span.className = "pill" + (cell.pill ? " esc" : "");
          span.textContent = cell.t;
          td.appendChild(span);
        } else {
          td.textContent = cell.t;
        }
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  ["before", "after"].forEach(function (id) {
    $(id).addEventListener("input", render);
  });
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
