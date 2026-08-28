# LossBench — Implementation Plan (P0 → P3)

Status: ratified
Date: 2026-08-14
Executor: 30 parallel agents, orchestrated from this document and the contract registry.

## 0. Status log

| Wave | Packages | Status |
|---|---|---|
| P0 | Foundation (contract registry, metrics, decision core, cost profiles, theorem tests) | DONE — 284 tests green, lint clean |
| P1 wave 1 | P1.1 generator, P1.2 contamination, P1.3 cache, P1.4 calibrate, P1.5 features, P1.6 policy, P1.7 runners (+P1.20 baselines), P1.8 record, P1.9 CLI, P1.10 ledger, P1.11 TPS, P1.12 pass^k, P1.13 sensitivity, P1.14 registry data | DONE |
| P1 wave 2 | P1.15 payment-repair generator, P1.16 settlement generator, P1.17 report, P1.18 HF packaging, P1.19 determinism utils | DONE |
| P2 | P2.1 eval harness, P2.2 replay lab, P2.3 calibrate pipeline, P2.4 HITL, P2.5 langgraph adapter, P2.6 dsh adapter, P2.7 frontier report, P2.8 server | DONE |
| Fragments | CLI `simulate` delegated to `ReplayLab` (was a placeholder re-implementing replay math); `generate_suite` dispatches all three domains (reconciliation/payment_repair/settlement); integration test suite added | CLOSED |
| Docs | `docs/ARCHITECTURE.md` (diagrams, module map, contracts), full README | DONE |
| P2.9 | Drift monitor (PSI on expected-loss distribution -> fail-safe escalate; KS on calibrated p -> recalibrate; realized ECE delta) — design claim C4 | DONE |
| P2.10 | Buzz collaboration projection (outbox, idempotent enqueues, verified resolution callback, payload builder) | DONE |
| P3 | Leaderboard Space, interactive demo Space (flagship replay), fine-tune scaffolding (export + mlx skeleton), full-run script + real artifacts, launch blog post draft, arXiv paper draft | DONE |

**Evidence rule:** nothing goes on a model/dataset card unless it is regenerable from a tagged repo state plus a committed artifact (`artifacts/<run_id>/`, tag `results-v0.1.0` pattern).

### Contract amendments (additive, ratified 2026-08-14)

1. `AuditLedger.read_all(limit: int = 1000) -> list[DecisionEvent]` — bulk append-order reader (needed by P2.2/P2.3).
2. Severity convention: `DecisionEvent.risk_features` is float-typed by contract, so the severity of an event lives in `observed_outcome["severity"]` (dict[str, Any]); consumers fall back to `LOW`.
3. `risk_features["calibrated_p"]` is the canonical calibrated-risk key for policy decisions (P1.6 contract). CLI `decide` delegates to `PolicyEngine` and never re-implements policy logic.
4. Stub runner lookups: `task_id` param takes precedence over exact prompt match (generated tasks share one prompt).
5. `make install` injects `sitecustomize.py` into the venv: hardened python>=3.11.14 (uv-managed) skips `.pth` files with the macOS hidden flag that uv sets, silently breaking editable installs. sitecustomize is processed at startup and immune to uv re-syncs.
6. CLI `simulate` delegates to the canonical `ReplayLab` + `fit_escalation_threshold`; total cost includes review cost (`escalate_cost` per escalated case), so `before`/`after` are total-cost figures, not business-loss-only. Trace records may carry legacy top-level `error`/`severity`; they are folded into `observed_outcome` on load.
7. `generate_suite` dispatches all three domains; domain gold dicts are full verifier outcomes (payment_repair gold carries `repair`/`repair_safe`; settlement gold carries `exposure_class`).
8. Task IDs are domain-prefixed (`{domain}:{seed}:{index}`) so suites from different domains share one ledger without event-id collisions.
9. Harness event/trace/trajectory IDs include the model id (`evt-{model}-{task}-{seed}-{step}`); trials run with `seed + trial` so multi-model, multi-trial workloads fit one audit ledger.
10. `DriftMonitor` (P2.9) semantics: PSI on expected-loss distribution alerts `fail_safe_escalate`; KS on calibrated p and realized-ECE delta alert `recalibrate`.
11. Buzz projection (P2.10) is network-free by design: the outbox stores signed-event payloads; `mark_published` simulates delivery; `resolve_callback` validates shape + published state (ReviewService owns ledger writes).
12. SOTA review wave (10 agents, 2026-08-16) — fixes applied:
    - Signature spaces unified: generator and contamination monitor hash the same canonical dict (stored `task.signature` == monitor computation).
    - Engine rule 3 is severity-aware: ESCALATE iff `p >= tau` OR `p·K(σ) > escalate_cost` (Bayes guard; the flagship fix — a CRITICAL case at p=0.3 escalates). ROUTE `expected_loss` now includes the routed tier cost.
    - Harness events carry `observed_outcome.severity`, the `error` marker, and `risk_features.calibrated_p` — the benchmark→control pipeline is no longer severed.
    - CLI `simulate`: reads the canonical `risk_features['calibrated_p']` fallback, reports line numbers on validation errors, skips uncalibrated rows, no double-parse; dead `_event_fields` deleted.
    - ReplayLab: DENY-decided events carry no business loss (denied actions never execute); per-case diffs show actual decisions, not "AUTO".
    - `missed_high_loss_rate` denominator gated on errors; sensitivity analysis never counts exact ties as crossovers.
    - AuditLedger: `UNIQUE(seq)` + BEGIN/COMMIT/ROLLBACK transaction + per-ledger lock; `export_jsonl` emits chain fields for external verification.
    - Response cache hit path synthesizes deterministic token usage (warm == cold events).
    - Calibration pipeline: held-out ECE split (in-sample leak closed; `calibrated_ece_fit` + held-out `calibrated_ece` reported); events without p excluded from fits; `_resolve_severity` reads only the ratified `observed_outcome` source.
    - Server: 409 on duplicate events, 400 on tenant mismatch.
    - dsh bridge: resolution event id `{decision_id}@resolved`, AMEND→VERIFY (matches ReviewService), tool args can no longer override the tool name.
    - LangGraph adapter rewritten to the real AgentMiddleware surface (`name`, `wrap_model_call`, `wrap_tool_call`); single event per decision point; messages hashed, never persisted raw.
    - `make determinism` gate + `scripts/check_determinism.py` (byte-identical modulo runtime metadata); scipy declared; `.env.example`; `artifacts/` gitignored.
13. Known gaps (documented, not defects): HTTP service has no auth (dev-mode; `x-lossbench-key` is the first pilot gate); redaction policy for raw proposed_action/observed_outcome is config-only (mandatory before real traffic); drift→recalibration loop and MAPIE conformal guarantees are specced but not wired; OTel emission uses bespoke attrs rather than full GenAI semantic conventions; netcal declared but hand-rolled scipy/sklearn used; cache has no TTL; schema versioning/WAL policy not yet implemented; the dsh manifest is a bridge contract awaiting the JS shim (real cordis patch format to follow).

## 0. How this plan works

- **P0 is sequential** (one agent): it creates the contract registry + repo scaffold. Nothing else may start until P0 merges.
- **P1 is the big fan-out wave**: every package depends ONLY on P0 contracts. Up to 30 agents run in parallel, each owning a disjoint set of files.
- **P2 is integration**: packages combine P1 outputs. Parallelizable in groups of 2–4 once their P1 dependencies land.
- **P3 is release**: distribution artifacts. Parallelizable 3–5.
- Every stage has: **public API signatures** (must match exactly), **files owned** (no two packages touch the same file), and **acceptance criteria** (named tests that must pass).

Rules for every agent:

1. Only edit files listed in your package's "Files owned" section.
2. Import contracts ONLY from the modules listed in "Depends on" (never re-declare types).
3. Do not modify `pyproject.toml` unless your package spec says so.
4. Do not install packages not listed in your spec's "Dependencies".
5. Run `uv run pytest tests/<your tests>` before reporting done. Report: files created, tests passing, any contract deviations.
6. No comments in code unless the package spec explicitly requests documentation strings (docstrings on public functions are mandatory; inline comments forbidden).

## 1. Contract registry (created in P0 — stable)

All modules under `src/lossbench/`. These are the ONLY cross-package interfaces.

| Module | Exports |
|---|---|
| `src/lossbench/schema.py` | `DecisionKind`, `Severity`, `DecisionEvent`, `Task`, `DecisionRequest`, `DecisionResponse`, `CostProfile`, `CostSource`, `PolicyBundle` |
| `src/lossbench/metrics/loss.py` | `severity_weighted_loss`, `total_policy_loss`, `regret`, `expected_decision_cost` |
| `src/lossbench/metrics/coverage.py` | `risk_coverage_curve`, `loss_at_fixed_budget` |
| `src/lossbench/metrics/calibration.py` | `ece`, `reliability_curve`, `brier_score` |
| `src/lossbench/metrics/deferral.py` | `escalation_precision_recall`, `ask_f1`, `missed_high_loss_rate` |
| `src/lossbench/decision.py` | `bayes_route`, `escalate_iff`, `expected_escalation_gain` |
| `src/lossbench/costs/registry.py` | `load_cost_profile`, `list_cost_profiles`, `ProfileId` |
| `src/lossbench/costs/profiles/*.yaml` | `flat.yaml`, `reconciliation.yaml`, `principal_risk.yaml`, `review_heavy.yaml` |

Semantic rules:

- `K(σ)` = `CostProfile.severity_costs[σ.value]` (error cost of the failure at severity σ).
- `p̂_t` = calibrated probability of the relevant failure (float 0..1).
- Regret is always relative to a baseline policy; the baseline is a parameter, never a global.

## 2. Repo layout (final)

```
regretbench/
├── pyproject.toml            (P0)
├── Makefile                  (P0)
├── README.md                 (P0 skeleton, P3 content)
├── docs/IMPLEMENTATION.md    (this file)
├── src/lossbench/
│   ├── schema.py             (P0)
│   ├── decision.py           (P0)
│   ├── metrics/              (P0 core; P1 extends)
│   ├── costs/                (P0 registry; P1 profiles)
│   ├── generate/             (P1.1 reconciliation generator)
│   ├── contamination/        (P1.2)
│   ├── cache/                (P1.3)
│   ├── calibrate/            (P1.4)
│   ├── features/             (P1.5)
│   ├── policy/               (P1.6)
│   ├── runners/              (P1.7)
│   ├── record/               (P1.8)
│   ├── cli/                  (P1.9)
│   ├── ledger/               (P1.10)
│   ├── scoring/              (P1.11 TPS, P1.12 pass^k)
│   ├── eval/                 (P2.1)
│   ├── replay/               (P2.2)
│   ├── hitl/                 (P2.4)
│   ├── adapters/             (P2.5 langgraph, P2.6 dsh)
│   └── report/               (P2.7)
└── tests/
```

## 3. Stage P0 — Foundation (sequential, one agent)

Outcome: repo compiles, contract registry complete, theorem test green.

| ID | Deliverable | Files owned |
|---|---|---|
| P0.1 | `pyproject.toml` (uv, src layout, Python ≥3.11, deps: pydantic>=2, numpy, pyyaml, pytest, ruff), `Makefile` (`make validate` = ruff + pytest), `README.md` skeleton | `pyproject.toml`, `Makefile`, `README.md`, `.gitignore`, `.python-version` |
| P0.2 | `schema.py`: all contract types (pydantic v2). `DecisionKind` enum: ALLOW, ROUTE, VERIFY, ABSTAIN, ESCALATE, DENY. `Severity` enum: LOW, MEDIUM, HIGH, CRITICAL. `DecisionEvent`, `Task`, `DecisionRequest`, `DecisionResponse`, `CostSource`, `CostProfile` (fields per design spec §9), `PolicyBundle` | `src/lossbench/schema.py` |
| P0.3 | `metrics/loss.py`, `metrics/coverage.py`, `metrics/calibration.py`, `metrics/deferral.py` — pure functions, zero I/O | `src/lossbench/metrics/__init__.py`, `loss.py`, `coverage.py`, `calibration.py`, `deferral.py` |
| P0.4 | `decision.py`: `bayes_route`, `escalate_iff`, `expected_escalation_gain` | `src/lossbench/decision.py` |
| P0.5 | `costs/registry.py` + 4 YAML profiles | `src/lossbench/costs/__init__.py`, `registry.py`, `profiles/*.yaml` |
| P0.6 | Tests: `test_schema.py`, `test_metrics.py`, `test_decision.py`, `test_cost_profiles.py`, `test_flat_cost_theorem.py` (H0 executable assertion), `test_property_escalation.py` | `tests/*` |

### P0.2 schema.py — exact API

```python
class DecisionKind(str, Enum): ALLOW ROUTE VERIFY ABSTAIN ESCALATE DENY
class Severity(str, Enum): LOW MEDIUM HIGH CRITICAL

class DecisionEvent(BaseModel):
    event_id: str
    tenant_id: str = "default"
    trace_id: str
    trajectory_id: str
    task_id: str
    parent_event_id: str | None = None
    timestamp: datetime
    input_snapshot_hash: str
    prompt_hash: str
    model_id: str
    model_revision: str = ""
    harness_id: str = ""
    harness_revision: str = ""
    reasoning_effort: str | None = None
    tool_name: str | None = None
    proposed_action: dict[str, Any] | None = None
    observed_outcome: dict[str, Any] | None = None
    risk_features: dict[str, float] = {}
    calibrated_probability: float | None = None
    expected_loss: float | None = None
    decision: DecisionKind
    rationale: str = ""
    policy_id: str
    policy_revision: str = ""
    cost_model_id: str
    token_usage: dict[str, int] = {}
    latency_ms: float = 0.0
    model_cost: float = 0.0
    judge_cost: float = 0.0
    human_cost: float = 0.0
    evidence_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

class Task(BaseModel):
    id: str
    domain: str
    prompt: str
    initial_state: dict[str, Any]
    available_tools: list[str]
    policy_id: str
    gold: dict[str, Any]
    severity: Severity
    verifier: str
    cost_model_ref: str
    difficulty: float = 0.5
    seed: int
    signature: str = ""

class DecisionRequest(BaseModel):
    tenant_id: str
    task_type: str
    trajectory_state: dict[str, Any]
    proposed_action: dict[str, Any]
    risk_features: dict[str, float]
    available_models: list[str]
    budget_state: dict[str, float] = {}
    evidence_refs: list[str] = []
    policy_ref: str

class DecisionResponse(BaseModel):
    decision: DecisionKind
    selected_model: str | None = None
    reasoning_effort: str | None = None
    requires_human: bool = False
    expected_loss: float | None = None
    confidence: float | None = None
    rationale: str = ""
    policy_ref: str = ""
    evidence_requirements: list[str] = []
    expires_at: datetime | None = None

class CostSource(BaseModel):
    title: str
    url: str
    date: str
    note: str = ""

class CostProfile(BaseModel):
    id: str
    description: str
    version: str = "0.1.0"
    sources: list[CostSource] = []
    severity_costs: dict[str, float]  # key = Severity.value
    escalate_cost: float = 1.0
    judge_cost: float = 0.0
    latency_penalty_per_s: float = 0.0
    model_cost_per_1k_out_tokens: dict[str, float] = {}
```

### P0.3 metrics — exact API

```python
def severity_weighted_loss(
    errors: Sequence[bool], severities: Sequence[Severity], profile: CostProfile
) -> float
def total_policy_loss(
    errors, severities, profile, model_cost: float, judge_cost: float, human_cost: float
) -> float
def regret(realized: float, baseline: float) -> float
def expected_decision_cost(p: float, severity: Severity, profile: CostProfile) -> float
def risk_coverage_curve(
    probs: Sequence[float], errors: Sequence[bool],
    severities: Sequence[Severity], profile: CostProfile,
    n_points: int = 20,
) -> list[dict[str, float]]
def loss_at_fixed_budget(curve: list[dict[str, float]], budget: float) -> float
def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> dict
def reliability_curve(confidences, correct, n_bins=10) -> list[dict]
def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float
def escalation_precision_recall(
    escalated: Sequence[bool], should_escalate: Sequence[bool]
) -> dict[str, float]
def ask_f1(question_precision: Sequence[float], blocker_recall: Sequence[float]) -> dict[str, float]
def missed_high_loss_rate(
    errors: Sequence[bool], severities: Sequence[Severity],
    profile: CostProfile, escalated: Sequence[bool],
) -> float
```

### P0.4 decision.py — exact API

```python
def expected_escalation_gain(
    p_avoidable_error: float, severity: Severity,
    profile: CostProfile, judge_cost: float = 0.0,
) -> float
def escalate_iff(
    p_avoidable_error: float, severity: Severity,
    profile: CostProfile, judge_cost: float = 0.0,
) -> bool
def bayes_route(
    p_error: dict[str, float],          # model_id -> P(error)
    severity: Severity, profile: CostProfile,
    model_cost: dict[str, float],       # model_id -> per-task cost
) -> tuple[str, float]                  # (best_model, expected_cost)
```

### P0.6 theorem tests

- `test_flat_cost_theorem.py`: with `flat.yaml` (all severity_costs equal), ranking models by `severity_weighted_loss` equals ranking by error rate. Exhaustive over 3 synthetic models, 3 seeds.
- `test_property_escalation.py`: for a fixed calibrated risk ordering, increasing `severity_costs[HIGH]` never decreases the optimal escalation rate (sweep K × 10, assert monotonicity of `escalate_iff` True-ratio on a fixed synthetic population).

## 4. Stage P1 — Fan-out wave (≤30 parallel agents)

All packages depend ONLY on P0 contracts. File ownership is disjoint. `pyproject.toml` may be touched ONLY by packages marked "may add deps".

| ID | Package | Files owned | May add deps | Depends on |
|---|---|---|---|---|
| P1.1 | Reconciliation task generator (seeded, verifier-as-oracle) | `src/lossbench/generate/` | yes (mlx? no — none) | P0.2, P0.5 |
| P1.2 | Contamination monitor | `src/lossbench/contamination/` | no | P0.2 |
| P1.3 | Response cache (DuckDB, byte-identical keys) | `src/lossbench/cache/` | yes (duckdb) | P0.2 |
| P1.4 | Calibration pipeline (netcal/sklearn) | `src/lossbench/calibrate/` | yes (netcal, scikit-learn) | P0.3 |
| P1.5 | Risk feature extractor | `src/lossbench/features/` | no | P0.2, P0.3 |
| P1.6 | Policy engine (YAML → decision fn, threshold fitting) | `src/lossbench/policy/` | yes (pyyaml already) | P0.2, P0.4 |
| P1.7 | Model runners (mlx / openai-compatible / vllm stub) | `src/lossbench/runners/` | yes (openai) | P0.2 |
| P1.8 | Trajectory recorder (OTel spans → DecisionEvent) + proxy mode | `src/lossbench/record/` | yes (opentelemetry) | P0.2 |
| P1.9 | CLI skeleton | `src/lossbench/cli/` | yes (click) | P0.2, P0.4, P0.5 |
| P1.10 | Audit ledger (Parquet/DuckDB store + hash chain) | `src/lossbench/ledger/` | yes (duckdb) | P0.2 |
| P1.11 | Trajectory Proper Score implementation | `src/lossbench/scoring/tps.py` | no | P0.2, P0.3 |
| P1.12 | Outcome-verified pass^k + false-success rate | `src/lossbench/scoring/passk.py` | no | P0.2, P0.3 |
| P1.13 | Cost-sensitivity analysis | `src/lossbench/metrics/sensitivity.py` | no | P0.3, P0.5 |
| P1.14 | Severity-cost registry data (empirical, sourced) | `src/lossbench/costs/registry_data.py` + `data/` | no | P0.5 |
| P1.15 | Payment-repair task generator | `src/lossbench/generate/payment_repair.py` | no | P0.2, P0.5, P1.1 interface |
| P1.16 | Settlement-risk task generator | `src/lossbench/generate/settlement.py` | no | P0.2, P0.5, P1.1 interface |
| P1.17 | Report generator (markdown/HTML) | `src/lossbench/report/` | no | P0.3, P1.13 |
| P1.18 | HF eval.yaml + dataset card packaging | `packaging/hf/` | no | P0.2 |
| P1.19 | Determinism utilities (seed policy, hash helpers) | `src/lossbench/util/determinism.py` | no | P0.2 |
| P1.20 | Multi-model comparison harness stub (deepseek/qwen baselines config) | `src/lossbench/runners/baselines.py` | no | P0.2, P1.7 |

### P1.1 generator — exact API (the pattern for P1.15/P1.16)

```python
# src/lossbench/generate/__init__.py
DOMAINS = ("reconciliation", "payment_repair", "settlement")

def generate_suite(
    domain: str, seed: int, n_tasks: int,
    severity_mix: dict[str, float] | None = None,
    difficulty: tuple[float, float] = (0.0, 1.0),
    verifier: Callable[[Task, dict[str, Any]], bool] | None = None,
) -> list[Task]
def verifier_reconciliation(task: Task, proposed_outcome: dict[str, Any]) -> bool
def task_signature(task: Task) -> str   # SHA-256 of sorted (field, value) pairs
```

Acceptance: same seed ⇒ byte-identical `Task` list (json round-trip). Every generated task passes its domain verifier (100% agreement). `task_signature` differs across distinct tasks. Severity mix honored within ±5% tolerance.

## 5. Stage P2 — Integration (parallel groups of 2–4)

| ID | Package | Depends on | Files owned |
|---|---|---|---|
| P2.1 | Agent-mode eval harness (trajectory runner, pass^k, checkpointing) | P1.1, P1.3, P1.12, P1.7 | `src/lossbench/eval/` |
| P2.2 | Replay lab (policy-only counterfactual simulator) | P1.10, P1.6, P1.5, P1.13 | `src/lossbench/replay/` |
| P2.3 | Calibrate+threshold pipeline (fits policy from ledger labels) | P1.4, P1.5, P1.6, P1.10 | `src/lossbench/calibrate/pipeline.py` |
| P2.4 | HITL review service (LangGraph interrupt + Temporal optional) | P1.10 | `src/lossbench/hitl/` |
| P2.5 | LangGraph adapter (middleware) | P1.6, P1.8 | `src/lossbench/adapters/langgraph.py` |
| P2.6 | DeepSeek Harness plugin (`dsh-plugin`) | P1.6, P1.8 | `src/lossbench/adapters/dsh/` |
| P2.7 | Cost-sensitivity frontier + report wiring | P1.13, P1.17 | `src/lossbench/report/frontier.py` |
| P2.8 | Multitenant decision HTTP service (FastAPI) | P1.6, P1.10 | `src/lossbench/server/` |

Acceptance for P2.2 (the flagship): load a recorded workload (Parquet of DecisionEvents), flip a policy threshold, re-emit all decisions deterministically (no LLM calls), output `{total_loss_before, total_loss_after, review_load_before, review_load_after, per_case_diff}`.

## 6. Stage P3 — Release (parallel 3–5)

| ID | Package | Depends on |
|---|---|---|
| P3.1 | HF leaderboard Space (Gradio) | P2.7, P1.18 |
| P3.2 | Interactive control-plane demo Space | P2.2, P2.4 |
| P3.3 | Model fine-tune runs (ReconForge 1.7B continuity; Qwen3.8-27B adapter candidate) | P2.1 baselines |
| P3.4 | Launch blog post + HF community post | all |
| P3.5 | arXiv credibility paper (TPS + severity-weighted trajectory loss) | P1.11, P2.7 |
| P3.6 | Full benchmark run + contamination certificate + model cards | P2.1, P2.7 |

## 7. Execution protocol (orchestrator)

1. Merge P0. Run `make validate` → green.
2. Launch P1 packages in one parallel wave (agents receive: this document, the P1 table row for their package, and the contract signatures from §3). Cap concurrency at 30; file-ownership check via `git status` before each agent merge.
3. On each merge: run `make validate`; if red, revert that agent's files only (its files are disjoint by construction) and relaunch.
4. Launch P2 groups as their P1 dependencies land; P3 in the final window.
5. Interface deviations are the only allowed re-merge reason: update the contract registry ONLY via a P0-owner commit, then notify dependents.
