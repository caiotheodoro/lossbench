"""Contract registry for LossBench.

This module is the single source of truth for cross-package types. All other
packages import from here and never re-declare these types.

Versioned contract: do not change field names or semantics without a P0-owner
commit and a notification to all dependents.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DecisionKind(StrEnum):
    """The intentionaly small decision enum emitted by the policy point."""

    ALLOW = "ALLOW"
    ROUTE = "ROUTE"
    VERIFY = "VERIFY"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"
    DENY = "DENY"


class Severity(StrEnum):
    """Business severity band for a task's failure mode."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DecisionEvent(BaseModel):
    """Append-only record of one decision point in an agent trajectory."""

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
    risk_features: dict[str, float] = Field(default_factory=dict)
    calibrated_probability: float | None = None
    expected_loss: float | None = None
    decision: DecisionKind
    rationale: str = ""
    policy_id: str
    policy_revision: str = ""
    cost_model_id: str
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    model_cost: float = 0.0
    judge_cost: float = 0.0
    human_cost: float = 0.0
    evidence_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class Task(BaseModel):
    """A benchmark task. Every task must pass its domain verifier."""

    id: str
    domain: str
    prompt: str
    initial_state: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)
    policy_id: str
    gold: dict[str, Any]
    severity: Severity
    verifier: str
    cost_model_ref: str
    difficulty: float = 0.5
    seed: int
    signature: str = ""


class DecisionRequest(BaseModel):
    """Input to the policy decision point."""

    tenant_id: str
    task_type: str
    trajectory_state: dict[str, Any] = Field(default_factory=dict)
    proposed_action: dict[str, Any]
    risk_features: dict[str, float] = Field(default_factory=dict)
    available_models: list[str] = Field(default_factory=list)
    budget_state: dict[str, float] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    policy_ref: str


class DecisionResponse(BaseModel):
    """Output of the policy decision point."""

    decision: DecisionKind
    selected_model: str | None = None
    reasoning_effort: str | None = None
    requires_human: bool = False
    expected_loss: float | None = None
    confidence: float | None = None
    rationale: str = ""
    policy_ref: str = ""
    evidence_requirements: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None


class CostSource(BaseModel):
    """Attribution for one empirical cost figure in a CostProfile."""

    title: str
    url: str
    date: str
    note: str = ""


class CostProfile(BaseModel):
    """Versioned, sourced mapping of severity -> business error cost.

    K(sigma) is the cost of a failure at that severity. All loss math reads
    from this profile; profiles are swappable inputs, never hidden constants.
    """

    id: str
    description: str
    version: str = "0.1.0"
    sources: list[CostSource] = Field(default_factory=list)
    severity_costs: dict[str, float]
    escalate_cost: float = 1.0
    judge_cost: float = 0.0
    latency_penalty_per_s: float = 0.0
    model_cost_per_1k_out_tokens: dict[str, float] = Field(default_factory=dict)

    def cost(self, severity: Severity) -> float:
        """K(sigma) for a severity enum value."""
        return self.severity_costs[severity.value]


class PolicyBundle(BaseModel):
    """A versioned policy: thresholds, model tier costs, escalation rules."""

    id: str
    revision: str = "0.1.0"
    cost_model_id: str
    escalation_threshold: float
    route_thresholds: dict[str, float] = Field(default_factory=dict)
    model_tiers: dict[str, float] = Field(default_factory=dict)
    spend_cap: float | None = None
    latency_sla_s: float | None = None
    allowlist: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
