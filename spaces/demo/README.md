---
title: LossBench Control Plane
emoji: 📉
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Re-run last month's workload under a different risk policy
---

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

Publishing is automated:

```sh
uv run python packaging/hf/publish_space.py --dry-run
uv run python packaging/hf/publish_space.py --repo caiotheodoro/lossbench-demo
```

It uploads `demo.py` as `app.py` (the name the front matter declares),
alongside `requirements.txt` and this README. The front matter above supplies
the SDK and version, so the Space resolves without any manual setup.

Dependencies come from `requirements.txt`: `lossbench` is not on PyPI, so it
installs from the GitHub repo. Pin a revision there before publishing so the
Space cannot drift with `main`.

To run locally: `uv run python spaces/demo/demo.py`, or run the same
functions headlessly via `build_workload` and `simulate_ui`.
