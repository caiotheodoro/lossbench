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
| `sample_leaderboard.json` | Default fixture (3 models, one with None optional fields) used as the Space default |

## Leaderboard JSON format

```json
{
  "models": [
    {"model_id": "reconforge-1.7b", "loss": 0.42, "pass_k": 0.87,
     "ece": 0.031, "escalated": 14, "total_cost": 92.5},
    {"model_id": "baseline-gpt-4o", "loss": 1.6, "escalated": 21, "total_cost": 210.0}
  ],
  "sensitivities": {
    "reconforge-1.7b": [{"ratio": 1.0, "loss": 0.42}, {"ratio": 10.0, "loss": 4.02}]
  },
  "honest_limits": ["Rankings depend on the severity-cost regime."]
}
```

- `models` is required; `loss` and `model_id` are required per entry, the rest
  are optional (absent -> `-` in the table).
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
