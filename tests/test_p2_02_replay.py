"""P2.2 replay lab acceptance tests: the flagship counterfactual demo."""

from __future__ import annotations

import sys
from datetime import datetime

import pytest

from lossbench.costs.registry import load_cost_profile
from lossbench.ledger.store import AuditLedger
from lossbench.replay.simulator import ReplayLab, ReplayOutcome
from lossbench.schema import DecisionEvent, DecisionKind, PolicyBundle, Severity


def _event(
    event_id: str,
    p: float,
    severity: Severity,
    error: bool,
    tenant: str = "default",
) -> DecisionEvent:
    outcome: dict = {"error": error, "severity": severity.value} if error else {
        "severity": severity.value
    }
    return DecisionEvent(
        event_id=event_id,
        tenant_id=tenant,
        trace_id="t",
        trajectory_id="tr",
        task_id="task",
        timestamp=datetime(2026, 8, 1),
        input_snapshot_hash="i",
        prompt_hash="p",
        model_id="m",
        calibrated_probability=p,
        risk_features={},
        observed_outcome=outcome,
        decision=DecisionKind.ALLOW,
        policy_id="pol",
        cost_model_id="reconciliation",
    )


def _workload() -> list[DecisionEvent]:
    high = [_event(f"h{i}", 0.9, Severity.HIGH, True) for i in range(10)]
    low = [_event(f"l{i}", 0.1, Severity.LOW, False) for i in range(10)]
    return high + low


def _policy(threshold: float) -> PolicyBundle:
    return PolicyBundle(
        id="pol", cost_model_id="reconciliation", escalation_threshold=threshold
    )


PROFILE = load_cost_profile("reconciliation")


def test_flip_threshold_changes_loss():
    lab = ReplayLab(PROFILE)
    outcome = lab.simulate(_workload(), _policy(0.5), new_threshold=0.95)
    # before: all 10 HIGH (p=0.9>=0.5) escalated -> 10 reviews, no business loss
    # after: nothing escalated -> 10 HIGH errors * K=10 = 100 loss
    assert outcome.before_loss < outcome.after_loss
    assert outcome.before_review_load == pytest.approx(0.5)
    assert outcome.after_review_load == pytest.approx(0.0)
    assert outcome.total_after > outcome.total_before
    # escalated-but-cheap cases: HIGH at p=0.9 with K=10 => escalation is right
    # (10 * escalate_cost = 10 vs 100 business loss)
    assert outcome.before_loss == pytest.approx(10.0)


def test_deterministic():
    lab = ReplayLab(PROFILE)
    a = lab.simulate(_workload(), _policy(0.5), 0.95)
    b = lab.simulate(_workload(), _policy(0.5), 0.95)
    assert a == b


def test_no_llm_calls():
    before = set(sys.modules)
    lab = ReplayLab(PROFILE)
    lab.simulate(_workload(), _policy(0.5), 0.95)
    after = set(sys.modules)
    new = after - before
    assert not any("runners" in m or "openai" in m for m in new)


def test_per_case_diff_lists_changed_events():
    lab = ReplayLab(PROFILE)
    outcome = lab.simulate(_workload(), _policy(0.5), 0.95)
    # events escalated before but not after: the 10 HIGH events
    changed_ids = {d["event_id"] for d in outcome.per_case_diff}
    assert changed_ids == {f"h{i}" for i in range(10)}
    for diff in outcome.per_case_diff:
        assert diff["before"] == "ESCALATE"
        assert diff["after"] == "ALLOW"


def test_empty_events():
    lab = ReplayLab(PROFILE)
    outcome = lab.simulate([], _policy(0.5), 0.95)
    assert outcome.before_loss == 0.0
    assert outcome.after_loss == 0.0
    assert outcome.before_review_load == 0.0
    assert outcome.per_case_diff == []


def test_review_load_fractions():
    lab = ReplayLab(PROFILE)
    outcome = lab.simulate(_workload(), _policy(0.95), 0.95)
    assert outcome.before_review_load == 0.0
    assert outcome.after_review_load == 0.0
    outcome2 = lab.simulate(_workload(), _policy(0.0), 0.0)
    assert outcome2.before_review_load == pytest.approx(1.0)


def test_simulate_with_ledger_matches_direct():
    ledger = AuditLedger()
    for event in _workload():
        ledger.append(event)
    lab = ReplayLab(PROFILE)
    direct = lab.simulate(_workload(), _policy(0.5), 0.95)
    via_ledger = lab.simulate_with_ledger(ledger, _policy(0.5), 0.95)
    assert direct == via_ledger


def test_severity_from_features():
    lab = ReplayLab(PROFILE)
    events = [
        _event("x", 0.05, Severity.HIGH, True),
        _event("y", 0.05, Severity.LOW, True),
    ]
    outcome = lab.simulate(events, _policy(0.9), 0.9)
    # both unescalated; HIGH error costs 10, LOW costs 0.2
    assert outcome.after_loss == pytest.approx(10.2)


def test_outcome_dataclass_shape():
    outcome = ReplayLab(PROFILE).simulate(_workload(), _policy(0.5), 0.95)
    assert isinstance(outcome, ReplayOutcome)
    assert set(outcome.per_case_diff[0]) == {
        "event_id",
        "before",
        "after",
        "expected_loss",
    }
