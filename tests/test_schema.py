from datetime import datetime

import pytest
from pydantic import ValidationError

from lossbench.schema import (
    CostProfile,
    CostSource,
    DecisionEvent,
    DecisionKind,
    DecisionRequest,
    DecisionResponse,
    PolicyBundle,
    Severity,
    Task,
)


def test_severity_enum_values():
    assert [s.value for s in Severity] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_decision_kind_values():
    assert [d.value for d in DecisionKind] == [
        "ALLOW",
        "ROUTE",
        "VERIFY",
        "ABSTAIN",
        "ESCALATE",
        "DENY",
    ]


def test_decision_event_roundtrip():
    event = DecisionEvent(
        event_id="e1",
        trace_id="t1",
        trajectory_id="tr1",
        task_id="task1",
        timestamp=datetime(2026, 8, 14, 12, 0, 0),
        input_snapshot_hash="abc",
        prompt_hash="def",
        model_id="qwen3.8-27b",
        decision=DecisionKind.ESCALATE,
        policy_id="pol1",
        cost_model_id="reconciliation",
    )
    data = event.model_dump()
    restored = DecisionEvent.model_validate(data)
    assert restored == event


def test_decision_event_defaults():
    event = DecisionEvent(
        event_id="e1",
        trace_id="t1",
        trajectory_id="tr1",
        task_id="task1",
        timestamp=datetime(2026, 8, 14),
        input_snapshot_hash="a",
        prompt_hash="b",
        model_id="m",
        decision=DecisionKind.ALLOW,
        policy_id="p",
        cost_model_id="flat",
    )
    assert event.tenant_id == "default"
    assert event.risk_features == {}
    assert event.created_at is not None


def test_decision_event_requires_policy_and_cost_model():
    with pytest.raises(ValidationError):
        DecisionEvent(
            event_id="e",
            trace_id="t",
            trajectory_id="tr",
            task_id="task",
            timestamp=datetime(2026, 8, 14),
            input_snapshot_hash="a",
            prompt_hash="b",
            model_id="m",
            decision=DecisionKind.ALLOW,
        )


def test_task_roundtrip():
    task = Task(
        id="task1",
        domain="reconciliation",
        prompt="Compare the two records.",
        gold={"verdict": "MATCH"},
        severity=Severity.HIGH,
        verifier="verifier_reconciliation",
        cost_model_ref="reconciliation",
        seed=7,
        policy_id="p1",
    )
    restored = Task.model_validate(task.model_dump())
    assert restored == task


def test_cost_profile_cost_accessor():
    profile = CostProfile(id="flat", description="d", severity_costs={"HIGH": 5.0})
    assert profile.cost(Severity.HIGH) == 5.0
    with pytest.raises(KeyError):
        profile.cost(Severity.CRITICAL)


def test_cost_source_attribution():
    source = CostSource(
        title="Fed Payments Study", url="https://example.com", date="2026-07"
    )
    assert source.note == ""


def test_decision_request_response():
    req = DecisionRequest(
        tenant_id="tenant-a",
        task_type="payment_repair",
        proposed_action={"tool": "post_adjustment", "amount": 1000},
        available_models=["cheap", "frontier"],
        policy_ref="pol-v1",
    )
    resp = DecisionResponse(
        decision=DecisionKind.ESCALATE,
        requires_human=True,
        rationale="high exposure",
        policy_ref=req.policy_ref,
    )
    assert resp.decision == DecisionKind.ESCALATE
    assert resp.requires_human


def test_policy_bundle():
    bundle = PolicyBundle(
        id="pol1", cost_model_id="reconciliation", escalation_threshold=0.7
    )
    assert bundle.revision == "0.1.0"
    assert bundle.model_tiers == {}
