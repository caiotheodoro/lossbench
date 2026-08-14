# LossBench control plane — interactive demo Space

The flagship "re-run last month" demo: replay a recorded reconciliation
workload under a different escalation threshold and read the counterfactual
cost — zero LLM calls, fully deterministic.

## What the demo shows

- A 200-task reconciliation workload generated once at startup (seeded, so
  every viewer sees the same workload), run through the EvalHarness with the
  gold-keyed stub runner (task_id lookup), with every event appended to an
  in-memory AuditLedger.
- Two sliders: last month's policy threshold (default 0.5) and the proposed
  new threshold (default 0.9), plus a cost-model dropdown covering the four
  shipped profiles (flat, reconciliation, principal_risk, review_heavy).
- Press **Replay**: `ReplayLab` re-decides every recorded event under the
  new threshold and renders before/after total cost, review load, and a
  dataframe of the per-case decisions that changed.

All replay math runs through `lossbench.replay.simulator.ReplayLab`; the
Space adds only presentation. No API keys, no network calls.

## How to publish

1. Install dependencies: `uv sync` at the repo root (gradio is a dev
   dependency).
2. Create a new Space at <https://huggingface.co/new-space> — SDK: Gradio,
   CPU hardware (free tier is plenty).
3. Upload the repo, pointing the Space's app file at
   `spaces/demo/demo.py` (or rename it to `app.py`).
4. Provide a `requirements.txt` with:

   ```text
   lossbench @ git+https://github.com/<org>/regretbench.git@main
   gradio>=6.24
   ```

5. Launch. Output is deterministic for every viewer.

To run locally: `uv run python spaces/demo/demo.py`, or run the same
functions headlessly via `build_workload` and `simulate_ui`.
