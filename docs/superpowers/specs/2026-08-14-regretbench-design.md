# LossBench v2 - expected-loss evaluation and control for agentic back-office systems

Status: design approved for implementation
Date: 2026-08-14

## 1. Executive Decision

Build a finance-back-office benchmark and control-plane project in two sequenced
phases:

1. **LossBench**: an open, reproducible benchmark for severity-weighted,
   trajectory-level expected loss in financial back-office agents.
2. **LossBench Control**: a plug-and-play runtime, CLI, and optional
   multitenant service that records agent decisions, estimates risk, routes or
   escalates work, and replays alternative policies offline.

The benchmark is the first public artifact and the long-term moat. The control
plane uses the same trajectory and decision contracts, so it is designed into
the benchmark rather than retrofitted later.

The project must not become another general agent framework, generic
observability product, or model zoo.

## 2. Thesis

Accuracy and token cost are insufficient objectives for production agents that
touch money. The useful objective is expected operational loss:

> A production agent should estimate the cost of being wrong, choose an
> execution strategy, stop before irreversible harm, escalate when appropriate,
> and produce evidence that the policy worked.

### 2.1 Falsifiable claims

- **C1 - Metric divergence:** severity-weighted expected loss over agent
  trajectories ranks models and routing policies differently from accuracy and
  cost-only routing.
- **C2 - Flat-cost theorem:** when all task severities have equal cost, loss
  ranking reduces to accuracy ranking.
- **C3 - Control benefit:** a calibrated loss-aware controller reduces realized
  loss at a fixed model/review budget compared with accuracy routing,
  confidence-only routing, and static escalation.
- **C4 - Shift behavior:** calibration and loss-distribution monitoring detect
  operational degradation before aggregate accuracy becomes materially worse.
- **C5 - Runtime integrity:** approval and escalation gates prevent unauthorized
  mutating actions across supported harness integrations.

The project does not claim that one universal cost model exists. Cost models are
versioned, sourced, replaceable inputs, and all conclusions must be shown across
published sensitivity ranges.

## 3. Why This Project Now

Current routing systems optimize price and benchmark quality. NVIDIA Switchyard
is the clearest example: its published viability formula accounts for judge cost
and the price gap, but not the consequence of an error. LangChain's recent
harness work demonstrates that middleware placement can materially change agent
quality without changing weights. Stripe Kai and managed-agent products show
that harness infrastructure is becoming commoditized.

The opportunity is therefore not another runtime. It is the neutral measurement
and control layer that works across runtimes.

The current open-model ecosystem strengthens this timing:

- **DeepSeek Harness (`dsh`)** is an MIT-licensed, plugin-oriented agent
  harness from DeepSeek AI. It is a developer-preview ecosystem with an
  explicit `dsh-plugin` discovery path.
- **Qwen3.8-27B** is an Apache-2.0 native multimodal model aimed at long-horizon
  coding and office workflows, with 262K native context and a documented path
  to 1M context. Its model card already uses Hugging Face evaluation metadata.
- **Qwen3.8-2.4T-A95B** provides an open frontier reference point.
- **Nemotron-3.5-Lightning** provides an agent-oriented open small-active-model
  reference point.

These releases make the distinction between model, harness, and control policy
more important. LossBench should integrate with each, not compete with them.

## 4. Product Shape

LossBench is one project with three public surfaces.

### 4.1 Benchmark and scoring engine

The category-defining artifact:

- finance-back-office tasks and agent environments
- verifier-as-oracle ground truth
- severity-cost registry and pluggable cost models
- trajectory-level scoring
- outcome-verified success and false-success detection
- calibrated deferral and human escalation metrics
- reproducible reports and leaderboard artifacts

### 4.2 Model family

Reference implementations, not the primary product:

- ReconForge 1.7B line: laptop/edge baseline and continuity of the published
  result
- Qwen3.8-27B finance back-office adapter: flagship open-weight model
- Nemotron-3.5-Lightning adapter: agent-oriented open-model comparison
- Qwen3.8-2.4T-A95B: open frontier comparison where inference is affordable
- DeepSeek V4-Flash: continuity baseline

The Qwen3.8 model card advertises office workflows and includes finance-related
agent benchmarks, making it the natural flagship base. The 1.7B model remains
important because it tests whether a small, local model can minimize financial
loss rather than maximize generic capability.

### 4.3 Control plane

LossBench Control exposes five integration surfaces:

1. **CLI** for zero- or low-code adoption
2. **Python library** for decorators and middleware
3. **OpenTelemetry/proxy integration** for harness-agnostic recording
4. **HTTP service** for multitenant policy evaluation and decisioning
5. **Collaboration projection** for human review and operational evidence

The first native adapters are:

- LangGraph/Deep Agents middleware
- DeepSeek Harness `dsh-plugin`

Other runtimes use the CLI, OpenAI-compatible proxy, or OpenTelemetry adapter.
Do not implement a custom runtime or fork an existing harness.

## 5. Formal Model

Let a trajectory be `tau = (s_0, a_1, s_1, ..., a_T, s_T)` and let `pi` be an
execution policy. A policy may select a model, reasoning effort, tool action,
verification step, abstention, or human escalation at each decision point.

For each decision point `t`:

- `K(sigma_t)` is the business cost of the applicable failure severity
- `p_hat_t` is a calibrated probability of the relevant failure
- `c_model_t` is inference cost, including reasoning tokens
- `c_judge_t` is the cost of an invoked judge or verifier
- `c_human_t` is human-review cost
- `c_latency_t` is an optional SLA/latency penalty
- `e_t` indicates an outcome error or unsafe action

```text
TotalLoss(pi, tau) = sum_t [
    K(sigma_t) * e_t
  + c_model_t
  + c_judge_t * judge_invoked_t
  + c_human_t * human_review_t
  + lambda * c_latency_t
]
```

Expected policy loss is measured over the task distribution:

```text
J(pi) = E_tau[TotalLoss(pi, tau)]
Regret(pi) = J(pi) - J(pi_star)
```

`pi_star` is an oracle or hindsight policy used only for evaluation. It is not
available to the live controller.

The controller's local Bayes decision is:

```text
route = argmin_m [ p_hat_m(x_t) * K(sigma_t) + price(m, effort) ]
```

Escalation is worthwhile when the expected avoided loss exceeds incremental
review and inference cost:

```text
escalate iff
  expected_avoided_loss(t)
  > incremental_model_cost(t)
  + judge_cost(t)
  + human_review_cost(t)
```

Judge cost must not be subtracted unconditionally. If a judge runs on every
request, that cost is common to all policies and cancels in a comparison. If it
runs conditionally, invocation probability must be included explicitly.

Qwen3.8's `reasoning_effort` is a first-class policy dimension:

```text
price(m, effort) = input_cost + output_cost + reasoning_token_cost(effort)
```

The controller can therefore choose low, medium, or high reasoning effort before
escalating to a different model.

## 6. Benchmark Scope

Version 1 is finance back-office only. It must not launch as a generic
multi-domain benchmark.

### 6.1 Tracks

#### Track A - Reconciliation

- ledger and statement pairs
- match/non-match decision
- exception classification
- severity classification
- evidence extraction
- escalation decision

Carry forward ReconForge's nine exception classes, verifier, contamination
signatures, and severity-weighted recall, but add trajectory and intervention
semantics.

#### Track B - Payment exception repair

- failed or returned payment triage
- repair proposal
- missing/contradictory field handling
- duplicate and replay detection
- approval before mutation
- final ledger-state verification

#### Track C - Settlement-risk escalation

- FX and value-date exceptions
- counterparty/beneficiary mismatch
- delayed or missing settlement
- exposure-sensitive severity
- mandatory human review for high-tail-risk actions

### 6.2 Agent tasks

Each task may require:

- retrieval from a controlled knowledge base
- structured tool calls
- ledger reads
- exception classification
- repair proposal
- verification
- escalation or abstention
- state mutation only after policy admission

The benchmark must evaluate the trajectory and resulting state, not only the
final natural-language answer.

### 6.3 Task contract

```text
Task {
  id,
  tenant_profile,
  prompt,
  initial_state,
  available_tools,
  policy,
  gold_outcome,
  severity,
  verifier,
  cost_model_ref
}
```

No task is accepted unless an independent verifier agrees with the generated
gold outcome. The verifier must not read the generator's intended label.

## 7. Metrics

The benchmark report must contain all of these; no single score is sufficient.

### 7.1 Primary metrics

- expected loss per trajectory
- normalized expected loss
- regret against always-frontier, always-cheap, and oracle policies
- loss at fixed spend budget
- loss at fixed human-review budget
- loss at fixed latency budget

### 7.2 Risk and deferral metrics

- severity-weighted risk-coverage curve
- escalation precision and recall
- missed-high-loss rate
- review load and review cost
- calibration ECE, Brier score, reliability curve
- Ask-F1-style escalation quality to prevent escalation spam

### 7.3 Agent integrity metrics

- outcome-verified pass@k and pass^k
- false-success rate
- unauthorized mutation rate
- policy-gate bypass rate
- state consistency after tool execution
- evidence completeness
- audit-record completeness

### 7.4 Research metrics

- trajectory proper score implementation
- cost-sensitivity curves
- ranking stability across severity ratios
- paired per-seed deltas and confidence intervals
- calibration degradation under shift


`K` is a versioned input, not a hidden benchmark constant.

The public registry contains:

- exception type
- operational interpretation
- severity band
- cost distribution or range
- source citation
- source date
- geography and business assumptions
- confidence level
- reviewer notes

Initial evidence sources include Fed payments statistics, ACH and wire
operational costs, chargeback fees, fraud-loss multipliers, Basel operational
risk event classes, settlement-risk publications, and human review time.

The benchmark ships multiple cost profiles:

- `flat.yaml` - theorem test and accuracy-equivalent baseline
- `reconciliation.yaml` - operational back-office costs
- `principal_risk.yaml` - large-value settlement emphasis
- `review_heavy.yaml` - high human-review cost
- user-supplied profiles

Every report must show where conclusions change as cost ratios vary.

## 9. Shared Runtime Contracts

The contracts are the most important design decision. They must be stable and
runtime-neutral.

### 9.1 Decision event

```text
DecisionEvent {
  event_id,
  tenant_id,
  trace_id,
  trajectory_id,
  task_id,
  parent_event_id,
  timestamp,
  input_snapshot_hash,
  prompt_hash,
  model_id,
  model_revision,
  harness_id,
  harness_revision,
  reasoning_effort,
  tool_name,
  proposed_action,
  observed_outcome,
  risk_features,
  calibrated_probability,
  expected_loss,
  decision,
  rationale,
  policy_id,
  policy_revision,
  cost_model_id,
  token_usage,
  latency_ms,
  model_cost,
  judge_cost,
  human_cost,
  evidence_hash,
  created_at
}
```

### 9.2 Decisions

The decision enum is intentionally small:

```text
ALLOW
ROUTE
VERIFY
ABSTAIN
ESCALATE
DENY
```

### 9.3 Policy input/output

```text
DecisionRequest {
  tenant_id,
  task_type,
  trajectory_state,
  proposed_action,
  risk_features,
  available_models,
  budget_state,
  evidence_refs,
  policy_ref
}

DecisionResponse {
  decision,
  selected_model,
  reasoning_effort,
  requires_human,
  expected_loss,
  confidence,
  rationale,
  policy_ref,
  evidence_requirements,
  expires_at
}
```

## 10. Plug-and-Play Surfaces

### 10.1 CLI

```sh
lossbench record -- command-to-run-agent
lossbench proxy --config tenant.yaml
lossbench decide --request request.json
lossbench simulate --trace traces.parquet --policy policy-v2.yaml
lossbench evaluate --suite finance-v1 --models config.yaml
lossbench report --run frontier-001
```

The CLI must work without LangGraph, DeepSeek Harness, or Temporal installed.

Optional Buzz review projection:

```sh
lossbench review publish --buzz-community https://relay.example --decision decision.json
lossbench review resolve --event resolution.json
```

### 10.2 Python library

```python
from lossbench import LossControl

control = LossControl.from_policy("tenant-policy.yaml")

decision = control.decide(
    task_type="payment_repair",
    proposed_action=action,
    risk_features=features,
)
```

The library is the canonical implementation. Framework adapters call it rather
than duplicating policy logic.

### 10.3 OpenTelemetry and proxy mode

Use OpenInference/OpenTelemetry for recording model and tool spans. Proxy mode
supports OpenAI-compatible clients and can capture requests without application
code changes. Raw prompts and sensitive inputs must be configurable for redaction
or hash-only storage.

### 10.4 Harness adapters

Launch adapters:

- `lossbench-langgraph`
- `lossbench-dsh`

The DeepSeek adapter is a plugin, not a fork. It should be discoverable through
the `dsh-plugin` topic and expose before-model, after-model, before-tool, and
after-tool hooks where the harness permits.

Additional runtimes are integration tests against the shared protocol, not
separate implementations.

Buzz ([block/buzz](https://github.com/block/buzz)) is an optional collaboration
projection, not another runtime dependency.
Its agent-as-member model, signed events, agent-first JSON CLI, ACP/MCP surface,
workflow approvals, search, and community-scoped tenancy are useful for the
human side of the control plane. LossBench can publish review requests,
evidence bundles, policy decisions, and human resolutions into a Buzz community
when configured. It must not fork or embed Buzz.

### 10.5 Buzz integration

The first integration is one-way for publication and verified for resolution:

```text
LossBench ledger -> outbox -> Buzz signed review event
Buzz resolution  -> verified callback -> LossBench ledger
```

LossBench remains canonical for loss calculations, policy versions,
calibration, replay, and release evidence. Buzz is the collaboration surface
where people and agents inspect a case, discuss evidence, approve or reject a
proposed action, and leave a durable signed resolution.

Map one LossBench `tenant_id` to one Buzz community boundary. Agent identities
must remain distinct from human identities. Review events should include:

- `decision_id` and `trajectory_id`
- policy and model revisions
- proposed action and expected loss
- redacted evidence references
- SLA deadline and required reviewer role
- resolution and resolution author

Use an outbox and idempotency key to avoid inconsistent dual writes. Signed
events do not provide confidentiality or regulatory retention by themselves;
use private/self-hosted communities, encryption or redaction, access controls,
and the LossBench retention policy for sensitive finance data.

## 11. Multitenancy

Multitenancy is part of the control-plane contract, not a premature hosted
product.

### 11.1 Isolation

- every record includes `tenant_id`
- tenant data is isolated at the database/query layer
- tenant secrets are never included in shared evaluation artifacts
- tenant-specific input and evidence are redacted before telemetry export
- tenant policy revisions are immutable
- tenant audit exports are independently verifiable

### 11.2 Tenant-specific configuration

Each tenant may define:

- cost model
- severity taxonomy mapping
- model allowlist
- reasoning-effort limits
- escalation SLAs
- review cost
- spend and latency budgets
- tool/action permissions
- retention and redaction policy
- calibration profile

### 11.3 Calibration hierarchy

Start with a global calibration model plus tenant-specific threshold. Add a
tenant-specific calibration model only after sufficient labeled outcomes exist.
Never silently fit a tenant model on the tenant's evaluation set.

### 11.4 Service modes

1. local embedded library
2. single-tenant self-hosted service
3. multitenant hosted decision service

Do not build hosted billing, dashboards, or enterprise administration before the
library and simulator have external users.

## 12. Control-Plane Architecture

```text
Agent or CLI
    |
    v
Recorder -> append-only event ledger -> feature/risk model
    |                                      |
    |                                      v
    |                              policy decision point
    |                              /    |       \
    v                             v     v        v
 model gateway              route  verify   escalate/deny
    |                                      |
    v                                      v
 model/tool execution              LangGraph interrupt or Temporal workflow
    |                                      |
    +------------ outcome and review -----+
                                           |
                                           v
                                 replay and calibration lab
```

Optional human collaboration projection:

```text
append-only ledger -> outbox -> Buzz community
       ^                              |
       +------ verified human resolution
```

Borrow rather than rebuild:

- OpenTelemetry/OpenInference for traces
- DuckDB/Parquet for replay analytics
- Postgres for append-only audit events
- LiteLLM or OpenAI-compatible gateway primitives
- LangGraph checkpoints and interrupts
- Temporal signals and durable timers for long-running review
- OPA/Rego or typed JSON policy bundles
- `netcal`, scikit-learn, and MAPIE for calibration/risk control
- Evidently or alibi-detect for drift
- MLX and Unsloth for fine-tuning
- vLLM/SGLang for serving Qwen3.8 when needed
- Gradio/Hugging Face Spaces for the public demo
- Buzz as an optional self-hosted collaboration and review surface

Build only the glue:

- trajectory schema
- response cache
- policy-only replay simulator
- cost-aware decision function
- audit evidence bundle
- framework adapters
- Buzz event projection and verified resolution callback

## 13. Model Strategy

### 13.1 Baseline matrix

Initial runs should include:

- ReconForge Qwen3-1.7B LoRA
- base Qwen3-1.7B
- Qwen3.8-27B zero-shot
- Qwen3.8-27B finance adapter
- Nemotron-3.5-Lightning
- DeepSeek V4-Flash
- Qwen3.8-2.4T-A95B where economically feasible

Use exact revisions and record inference engine, quantization, prompt, thinking
mode, reasoning effort, and sampling configuration.

### 13.2 Fine-tuning sequence

1. Reproduce ReconForge's 1.7B result.
2. Establish Qwen3.8-27B and Nemotron baselines before training.
3. Train the Qwen3.8 finance adapter on Kaggle or equivalent GPU.
4. Publish MLX/quantized variants only after the canonical result is stable.
5. Compare non-thinking, low, medium, and high reasoning effort as separate
   policy arms.

Qwen3.8's hybrid architecture and new tooling may make MLX/Unsloth support lag.
Do not make the 27B adapter a week-1 dependency. The 1.7B path preserves a
fully local fallback.

## 14. Reproducibility and Evaluation Integrity

- seeded task generation
- byte-identical generated suites
- SHA-256 contamination signatures
- frozen evaluation revisions
- exact model and harness revisions
- response cache keyed by model, prompt, parameters, seed, and input hash
- policy-only counterfactuals are deterministic and require no new LLM calls
- model/prompt counterfactuals use paired seeded runs
- report per-seed deltas, confidence intervals, and variance envelopes
- separate prediction artifacts from grading artifacts
- LLM judge agreement measured against human/oracle labels
- mandatory honest-limits and negative-results sections

The flat-cost theorem is an executable assertion. Additional property tests must
verify that increasing high-severity cost cannot reduce the optimal escalation
rate for the same calibrated risk ordering.

## 15. Hugging Face Release Package

Release as one publisher identity:

- benchmark dataset with `eval.yaml`
- model repositories with `.eval_results/`
- leaderboard Space
- interactive evaluation/control-plane Space
- dataset cards with task, license, contamination, and cost-model details
- MLX, GGUF, and supported quantized model variants
- `agents.md` describing machine-callable evaluation workflows
- MCP or HTTP endpoint for agent-triggered evaluation where practical
- HF community blog post
- arXiv credibility paper

The benchmark must become a dependency, not only a launch headline. Scores should
be machine-readable and reusable by other model cards.

## 16. Public Demo

The primary demo is:

> Re-run last month's back-office workload under a different risk policy.

The viewer shows:

- original decisions and model routes
- calibrated risk and severity
- policy and model revisions
- total expected and realized loss
- review load and spend
- a threshold/model/reasoning-effort slider
- deterministic counterfactual results
- per-case decision differences
- evidence bundles for escalated actions
- a drift event that triggers recalibration or fail-safe escalation

The demo must also show a live decision entering a human review workflow and
returning a durable, audited resolution.

## 17. Twelve-Week Build Plan

### Weeks 1-2: public stake

- publish this spec and metric definition
- implement pure metric library
- implement flat-cost theorem test
- create severity registry v0
- define trajectory and decision contracts
- publish generator sketch and H0/H1

Exit: public repository with tests and a reproducible example.

### Weeks 3-4: finance benchmark foundation

- reconciliation task suite
- independent verifier
- contamination monitor
- initial agent trajectory format
- 200+ deterministic tasks

Exit: verifier agreement 100%, contamination overlap 0.

### Weeks 5-6: scoring and calibration

- response cache
- severity-weighted loss curves
- calibrated risk features
- confidence and review thresholds
- MAPIE/conformal experiment

Exit: first cost-sensitive frontier on synthetic data.

### Weeks 7-8: agent tracks

- payment repair and settlement-risk tasks
- state mutation verification
- outcome-verified pass@k/pass^k
- false-success analysis
- trajectory proper score implementation

Exit: benchmark report identifies where accuracy ranking differs from loss ranking.

### Week 9: model release candidate

- baseline matrix
- Qwen3.8-27B adapter experiment
- 1.7B continuity run
- Nemotron comparison
- HF metadata preparation

Exit: third-party and self-owned baselines are reproducible.

### Weeks 10-11: thin control plane

- `lossbench decide`
- policy bundles
- local recorder and proxy
- LangGraph adapter
- DeepSeek Harness plugin
- offline policy simulator
- minimal Temporal review workflow

The Buzz review projection is an optional week-11 integration and is not a
launch blocker for the benchmark or core decision service.

Exit: threshold change replays recorded workload and emits a counterfactual
loss/review/spend report.

### Week 12: launch

- HF dataset and leaderboard
- interactive Space
- model cards
- arXiv draft
- technical launch post
- Switchyard issue/PR proposing loss-aware evaluation

Exit: an external user can run the benchmark or wrap an agent without reading
internal code.

## 18. Scope Boundaries

Do not build in v1:

- a new general-purpose agent runtime
- a generic observability dashboard
- a hosted model marketplace
- a broad healthcare or legal benchmark
- a large judge-model training project
- a full enterprise admin/billing console
- every framework adapter
- a claim that synthetic severity equals real financial loss

The control plane is a thin, protocol-first layer. The benchmark and evidence
artifacts are the priority.

## 19. Risks and Mitigations

### Arbitrary cost criticism

Use empirical registry, pluggable profiles, sensitivity curves, and explicit
confidence levels. Never publish only one chosen cost profile.

### Self-serving benchmark

Include at least five external models, publish all prompts and revisions, freeze
the test set, and publish negative results. The reference model is a baseline,
not proof of benchmark superiority.

### Benchmark contamination

Keep test data out of training and emit a contamination certificate with each
release. Do not train the flagship adapter on the public test set.

### Framework churn

Depend on the shared protocol and OTel. Treat LangGraph and `dsh` integrations as
thin adapters. Never make the benchmark depend on one harness.

### Qwen3.8 tooling immaturity

Run the 1.7B baseline and zero-shot 27B baseline first. Train the 27B adapter
only after an inference/training path is verified. Keep vLLM/SGLang as serving
fallbacks.

### Multitenant privacy

Hash or redact sensitive payloads, isolate tenant rows and keys, make retention
configurable, and never export private traces to public benchmark artifacts.

### Adopt AI intellectual property

Do not copy proprietary code, schemas, or customer data. Build the open control
plane from public interfaces and independently authored contracts. Confirm
employment and consulting agreements before extracting any implementation.

## 20. Success Criteria

### By week 2

- public spec and theorem test
- reproducible generator
- external reader can understand the metric

### By week 12

- benchmark runs locally from a clean checkout
- at least five model baselines
- outcome-verified agent track
- cost-sensitivity frontier
- HF dataset/leaderboard/Space
- LangGraph and `dsh` integration examples
- CLI can record and simulate a workload

### By twelve months

- external model or agent evaluated by another party
- benchmark cited or integrated by a model/harness project
- at least one finance operator or fintech team testing the control plane
- reproducible evidence that loss-aware policy beats cost-only routing at a fixed
  budget
- a credible decision between staff/lab adoption and product commercialization

## 21. Positioning

The project should be described as:

> The open expected-loss benchmark and control plane for agents that touch money.

Not:

- another LLM router
- another agent framework
- another observability dashboard
- another finance chatbot

The career and founder thesis is sequenced:

1. Own a trusted metric and benchmark.
2. Make it a dependency in the open-model ecosystem.
3. Make the same contracts plug into any agent harness.
4. Use benchmark evidence, auditability, and finance-specific cost models as the
   moat for a multitenant control-plane product.
