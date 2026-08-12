from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lossbench.hitl import ReviewResolution, ReviewService
from lossbench.hitl import review as review_module
from lossbench.ledger import AuditLedger
from lossbench.schema import DecisionEvent, DecisionKind


def make_event(
    event_id: str,
    decision: DecisionKind = DecisionKind.ESCALATE,
    trajectory_id: str = "traj-1",
    tenant_id: str = "tenant-a",
    expected_loss: float = 12.5,
) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        tenant_id=tenant_id,
        trace_id=f"trace-{event_id}",
        trajectory_id=trajectory_id,
        task_id=f"task-{event_id}",
        timestamp=datetime(2026, 8, 14, 12, 0, 0),
        input_snapshot_hash="in-1",
        prompt_hash="prompt-1",
        model_id="model-a",
        proposed_action={"type": "transfer", "amount": 1000},
        expected_loss=expected_loss,
        rationale="high risk amount",
        decision=decision,
        policy_id="policy-1",
        cost_model_id="cost-1",
    )


def test_open_review_requires_escalate():
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    with pytest.raises(ValueError):
        svc.open_review(make_event("dec-allow", decision=DecisionKind.ALLOW))
    request = svc.open_review(make_event("dec-1"))
    assert request.decision_id == "dec-1"
    assert request.trajectory_id == "traj-1"
    assert request.tenant_id == "tenant-a"
    assert request.task_id == "task-dec-1"
    assert request.proposed_action == {"type": "transfer", "amount": 1000}
    assert request.expected_loss == 12.5
    assert request.rationale == "high risk amount"
    assert request.policy_ref == "policy-1"
    assert request.sla_seconds == 28800
    assert request.required_role == "analyst"
    opened = ledger.get("dec-1@review-opened")
    assert opened is not None
    assert opened.decision == DecisionKind.ESCALATE
    assert opened.rationale.startswith("[review-opened]")
    assert opened.parent_event_id == "dec-1"
    assert opened.observed_outcome["decision_id"] == "dec-1"
    ledger.close()


def test_resolve_approve():
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    svc.open_review(make_event("dec-1"))
    resolved = svc.resolve(
        ReviewResolution(
            decision_id="dec-1", reviewer="alice", resolution="APPROVE", note="looks fine"
        )
    )
    assert resolved.decision == DecisionKind.ALLOW
    assert resolved.parent_event_id == "dec-1"
    assert resolved.observed_outcome["resolution"] == "APPROVE"
    stored = ledger.get("dec-1@review-resolved")
    assert stored is not None
    assert stored.decision == DecisionKind.ALLOW
    assert stored.parent_event_id == "dec-1"
    assert stored.observed_outcome["resolution"] == "APPROVE"
    ledger.close()


def test_resolve_reject_and_amend():
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    svc.open_review(make_event("dec-1"))
    rejected = svc.resolve(
        ReviewResolution(decision_id="dec-1", reviewer="bob", resolution="REJECT")
    )
    assert rejected.decision == DecisionKind.DENY
    with pytest.raises(ValueError):
        svc.resolve(
            ReviewResolution(decision_id="dec-1", reviewer="bob", resolution="AMEND")
        )
    svc.open_review(make_event("dec-2"))
    amended = svc.resolve(
        ReviewResolution(
            decision_id="dec-2",
            reviewer="bob",
            resolution="AMEND",
            amended_action={"type": "transfer", "amount": 500},
        )
    )
    assert amended.decision == DecisionKind.VERIFY
    assert amended.proposed_action == {"type": "transfer", "amount": 500}
    ledger.close()


def test_pending_reflects_resolutions():
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    svc.open_review(make_event("dec-1"))
    svc.open_review(make_event("dec-2"))
    svc.open_review(make_event("dec-3", tenant_id="tenant-b"))
    assert [r.decision_id for r in svc.pending()] == ["dec-1", "dec-2"]
    svc.resolve(
        ReviewResolution(decision_id="dec-1", reviewer="alice", resolution="APPROVE")
    )
    assert [r.decision_id for r in svc.pending()] == ["dec-2"]
    tenant_b = svc.pending("tenant-b")
    assert [r.decision_id for r in tenant_b] == ["dec-3"]
    assert tenant_b[0].tenant_id == "tenant-b"
    ledger.close()


def test_sla_overdue(monkeypatch):
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    base = datetime(2026, 8, 14, 9, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(review_module, "_utcnow", lambda: base)
    svc.open_review(make_event("dec-1"))
    assert svc.sla_overdue(now=base + timedelta(hours=7)) == []
    fast = make_event("dec-2")
    fast_open = DecisionEvent(
        event_id="dec-2@review-opened",
        tenant_id="tenant-a",
        trace_id=fast.trace_id,
        trajectory_id=fast.trajectory_id,
        task_id=fast.task_id,
        parent_event_id="dec-2",
        timestamp=base,
        input_snapshot_hash=fast.input_snapshot_hash,
        prompt_hash=fast.prompt_hash,
        model_id=fast.model_id,
        proposed_action=fast.proposed_action,
        observed_outcome={
            "decision_id": "dec-2",
            "sla_seconds": 1,
            "required_role": "analyst",
        },
        expected_loss=fast.expected_loss,
        decision=DecisionKind.ESCALATE,
        rationale="[review-opened] urgent",
        policy_id=fast.policy_id,
        cost_model_id=fast.cost_model_id,
        created_at=base,
    )
    ledger.append(fast_open)
    overdue = svc.sla_overdue(now=base + timedelta(seconds=2))
    assert [r.decision_id for r in overdue] == ["dec-2"]
    svc.resolve(
        ReviewResolution(
            decision_id="dec-2",
            reviewer="alice",
            resolution="APPROVE",
            resolved_at=base + timedelta(seconds=3),
        )
    )
    svc.resolve(
        ReviewResolution(
            decision_id="dec-1",
            reviewer="alice",
            resolution="APPROVE",
            resolved_at=base + timedelta(seconds=4),
        )
    )
    assert svc.sla_overdue(now=base + timedelta(hours=10)) == []
    ledger.close()


def test_deterministic_order(monkeypatch):
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    base = datetime(2026, 8, 14, 9, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(review_module, "_utcnow", lambda: base)
    for decision_id in ["dec-c", "dec-a", "dec-b"]:
        svc.open_review(make_event(decision_id))
    monkeypatch.setattr(review_module, "_utcnow", lambda: base + timedelta(seconds=1))
    svc.open_review(make_event("dec-d"))
    assert [r.decision_id for r in svc.pending()] == [
        "dec-a",
        "dec-b",
        "dec-c",
        "dec-d",
    ]
    late = base + timedelta(hours=10)
    assert [r.decision_id for r in svc.sla_overdue(now=late)] == [
        "dec-a",
        "dec-b",
        "dec-c",
        "dec-d",
    ]
    ledger.close()


def test_roundtrip_through_ledger(tmp_path: Path):
    path = str(tmp_path / "ledger.duckdb")
    ledger = AuditLedger(path)
    svc = ReviewService(ledger, tenant_id="tenant-a")
    svc.open_review(make_event("dec-1"))
    svc.open_review(make_event("dec-2"))
    svc.resolve(
        ReviewResolution(decision_id="dec-1", reviewer="alice", resolution="APPROVE")
    )
    ledger.close()
    reopened_ledger = AuditLedger(path)
    reopened = ReviewService(reopened_ledger, tenant_id="tenant-a")
    pending = reopened.pending()
    assert [r.decision_id for r in pending] == ["dec-2"]
    assert pending[0].expected_loss == 12.5
    assert pending[0].policy_ref == "policy-1"
    reopened_ledger.close()


def test_event_ids_unique_in_ledger():
    ledger = AuditLedger()
    svc = ReviewService(ledger, tenant_id="tenant-a")
    ledger.append(make_event("dec-1"))
    svc.open_review(make_event("dec-1"))
    svc.resolve(
        ReviewResolution(decision_id="dec-1", reviewer="alice", resolution="APPROVE")
    )
    original = ledger.get("dec-1")
    opened = ledger.get("dec-1@review-opened")
    resolved = ledger.get("dec-1@review-resolved")
    assert original is not None
    assert opened is not None
    assert resolved is not None
    assert {original.event_id, opened.event_id, resolved.event_id} == {
        "dec-1",
        "dec-1@review-opened",
        "dec-1@review-resolved",
    }
    assert ledger.verify()["valid"] is True
    assert ledger.verify()["n_events"] == 3
    ledger.close()
