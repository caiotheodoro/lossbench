# LossBench Leaderboard — Hugging Face Space (P3.1)

Gradio Space that renders the LossBench leaderboard from a **static JSON**
artifact — no live database, fully deterministic. It shows per-model
severity-weighted loss, pass^k, ECE, escalation counts and total cost, plus
the cost-sensitivity crossover summary and honest limits from the frontier
report.

## Files

| File | Purpose |
|---|---|
| `leaderboard.py` | Pure logic: `LeaderboardRow`, `load_leaderboard`, `render_table`, `crossover_summary` |
| `app.py` | Thin Gradio UI: `demo()` builds the Blocks; the `__main__` block launches it |
| `sample_leaderboard.json` | **Synthetic demo fixture** — invented `demo-model-a/b/c` numbers, carries a `banner: "SYNTHETIC DEMO DATA"` the UI renders as a loud warning. Never a real result, never a silent fallback. |

## Publish gate

This Space stays **unpublished** until the real full run (issue #23) produces a
non-partial leaderboard dataset. The committed `artifacts/leaderboard.json` is
stub pipeline smoke output (`"partial": true`, `"banner": "STUB PIPELINE SMOKE
OUTPUT"`) — every model scores perfectly because the stub is gold-keyed, so it
is not a result. This fix (issue #2) closes the fabrication hole — the loader
reads the real `severity_weighted_loss` key and any non-result data renders a
visible banner — but it does not lift the publish gate.

## Leaderboard JSON format

```json
{
  "models": [
    {"model_id": "demo-model-a", "severity_weighted_loss": 0.42, "pass_k": 0.87,
     "ece": 0.031, "escalated": 14, "total_cost": 92.5},
    {"model_id": "demo-model-b", "severity_weighted_loss": 1.6, "escalated": 21, "total_cost": 210.0}
  ],
  "sensitivities": {
    "demo-model-a": [{"ratio": 1.0, "loss": 0.42}, {"ratio": 10.0, "loss": 4.02}]
  },
  "honest_limits": ["Rankings depend on the severity-cost regime."],
  "banner": "SYNTHETIC DEMO DATA"
}
```

- `models` is required; `model_id` and `severity_weighted_loss` (the key the run
  artifacts and `scripts/full_run.py` emit; the legacy `loss` alias is still
  read) are required per entry, the rest are optional (absent -> `-` in the table).
- `banner` is optional; when present the UI renders it as a loud warning that
  the data is not a real result.
- `sensitivities` and `honest_limits` are optional and come straight from the
  frontier report; see `src/lossbench/report/frontier.py`
  (`frontier_report(...)` returns `{"losses", "sensitivities", "calibration",
  "deferral", "honest_limits"}`) — map `losses` to `models` entries and embed
  the report's `sensitivities`/`honest_limits` verbatim.

## Run locally

```bash
uv run python spaces/leaderboard/app.py
```

## Publish to Hugging Face

1. `huggingface-cli login`
2. Create the Space: `huggingface-cli repo create lossbench-leaderboard --type space` (or create it on the Hub and select **Gradio** as the SDK).
3. Push these three files plus a Space `README.md` (`sdk: gradio`, `sdk_version: 6.x`):

```bash
huggingface-cli upload LossBench/lossbench-leaderboard spaces/leaderboard . .
```

4. Upload the release leaderboard JSON (produced from `frontier_report`) to
   the Space's files so visitors can load it, or push it to the dataset repo
   (see below) and reference the raw URL.

## Wiring: dataset repo, eval.yaml, .eval_results

The leaderboard exists to surface results gathered through the HF Community
Evals flow: models registered against the LossBench dataset get scored and the
`severity_weighted_loss` metric lands on their model cards via `.eval_results`.
See **`packaging/hf/README.md`** for the full procedure:

- dataset repo `LossBench/lossbench` with `eval.yaml` (Community Evals
  registration), `data/eval.jsonl`, and `results/eval-results.json`;
- `.eval_results` on each model card, produced by the community eval runner
  (verifier-as-oracle, pass^k, severity-weighted loss across a cost-ratio
  range);
- export a run's results into this Space's JSON (`{"models": [...]}` with
  `sensitivities` + `honest_limits`) and re-upload to refresh the leaderboard.
