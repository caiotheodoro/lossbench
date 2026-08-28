# LossBench

**Expected-loss evaluation and control for agents that touch money.**

Accuracy and token cost are the wrong objectives for production agents. The
useful objective is expected operational loss: severity-weighted, calibrated,
auditable. LossBench is the open benchmark that scores it and the control
plane that acts on it — with the same trajectory contracts on both sides.

- **Evaluation half**: seeded finance back-office benchmarks (reconciliation,
  payment repair, settlement risk) with verifier-as-oracle ground truth,
  severity-weighted loss, cost-sensitivity analysis, and outcome-verified
  scoring (pass^k, Trajectory Proper Score, false-success detection).
- **Control half**: record -> calibrate -> decide -> escalate -> replay.
  Library, CLI, HTTP service, LangGraph middleware, and DeepSeek Harness
  plugin bridge, all over an append-only, hash-chained audit ledger.

```
LossBench Control exposes five integration surfaces:
  CLI        lossbench record/decide/simulate/evaluate/report
  Library    from lossbench import PolicyEngine, LossGuardMiddleware
  OTel/proxy OpenAI-compatible capture, zero code change
  Service    POST /v1/decide (multitenant, policy-isolated)
  Adapters   lossbench-langgraph middleware, lossbench-dsh plugin
```

## Install

```sh
make install          # uv sync + env fixes (see Makefile)
make validate         # ruff + 406 tests
```

## Quick start

```python
from lossbench.generate import generate_suite
from lossbench.costs.registry import load_cost_profile
from lossbench.schema import PolicyBundle, DecisionRequest
from lossbench.policy import PolicyEngine

tasks = generate_suite("reconciliation", seed=7, n_tasks=500)  # deterministic
profile = load_cost_profile("principal_risk")
policy = PolicyBundle(id="p1", cost_model_id="principal_risk", escalation_threshold=0.6)
engine = PolicyEngine(policy, profile)

decision = engine.decide(DecisionRequest(
    tenant_id="acme-bank",
    task_type="reconciliation",
    proposed_action={"tool": "post_settlement"},
    risk_features={"calibrated_p": 0.91},
    available_models=["reconforge-1.7b", "qwen3.8-27b"],
    policy_ref="p1",
))
# decision: ESCALATE, requires_human=True
```

## CLI

```sh
lossbench version
lossbench costs list
lossbench metrics check < results.jsonl          # loss + ECE summary
lossbench decide --request req.json --policy pol.yaml --cost-model reconciliation
lossbench simulate --trace traces.jsonl --policy pol.yaml --cost-model reconciliation
#   {"before": 50.0, "after": 5.0, "review_load_before": 0.0, "review_load_after": 0.5}
```

## The flagship demo

**"Re-run last month under a different risk policy."**

1. Every decision lands in the audit ledger as a `DecisionEvent`
   (model, prompt hash, calibrated risk, expected loss, policy revision, cost).
2. `ReplayLab.simulate(events, policy, new_threshold)` re-decides the whole
   workload deterministically — zero LLM calls.
3. Read the counterfactual: total loss, review load, per-case diffs with
   evidence bundles.

```python
from lossbench.ledger import AuditLedger
from lossbench.replay import ReplayLab
from lossbench.costs.registry import load_cost_profile

ledger = AuditLedger("workload.duckdb")
lab = ReplayLab(load_cost_profile("reconciliation"))
outcome = lab.simulate_with_ledger(ledger, policy, new_threshold=0.4)
print(outcome.total_before, outcome.total_after, len(outcome.per_case_diff))
```

## Artifacts

Each `scripts/full_run.py` invocation writes a self-contained evidence tree to
`artifacts/<run_id>/` (`run_id` = date + runner + seed + max_steps): its own
`leaderboard.json`, `report.md`, `contamination_certificate.json`,
`model_cards/`, `runconfig.json` (seed / runner / models / git revision), and
`workload.duckdb`. Stub runs carry an unmissable banner in every file. Runs
whose numbers are published are pinned to a local git tag (`results-v0.1.0`);
nothing goes on a card unless it is regenerable from a tagged state plus a
committed artifact.

## Architecture & docs

| Doc | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | System diagrams (Mermaid), data flows, module map, contracts, deployment modes |
| `docs/IMPLEMENTATION.md` | Staged build plan P0->P3, status log, contract amendments |
| `docs/superpowers/specs/2026-08-14-regretbench-design.md` | Full design: thesis, formal model, cost registry, HF release package |
| `packaging/hf/README.md` | HF dataset + Community Evals registration procedure |
| `packaging/dsh/README.md` | DeepSeek Harness plugin publication |

## The core claim

> Severity-weighted expected loss over agent trajectories ranks models and
> policies differently from accuracy — and the divergence grows with cost
> asymmetry. When severity is flat, loss ranking reduces to accuracy ranking
> (an executable theorem test). Cost models are versioned, sourced, swappable
> inputs; every conclusion is shown across a K range.

## License

Apache 2.0. See `LICENSE`.
