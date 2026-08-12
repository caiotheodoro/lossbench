from datetime import datetime

import numpy as np
import pytest

from lossbench.calibrate.pipeline import (
    fit_policy_from_ledger,
    run_calibration_pipeline,
)
from lossbench.ledger import AuditLedger
from lossbench.schema import CostProfile, DecisionEvent, DecisionKind, Severity


def _cost_profile() -> CostProfile:
    return CostProfile(
        id="flat",
        description="flat test profile",
        severity_costs={
            Severity.LOW.value: 1.0,
            Severity.MEDIUM.value: 5.0,
            Severity.HIGH.value: 25.0,
            Severity.CRITICAL.value: 100.0,
        },
        escalate_cost=1.0,
    )


def make_event(
    event_id: str,
    confidence: float,
    error: bool | None,
    severity: str | None = None,
) -> DecisionEvent:
    outcome = None if error is None else {"error": error}
    if severity is not None:
        outcome = dict(outcome or {})
        outcome["severity"] = severity
    return DecisionEvent(
        event_id=event_id,
        trace_id=f"trace-{event_id}",
        trajectory_id="traj-1",
        task_id=f"task-{event_id}",
        timestamp=datetime(2026, 8, 14, 12, 0, 0),
        input_snapshot_hash="in",
        prompt_hash="ph",
        model_id="m",
        calibrated_probability=confidence,
        observed_outcome=outcome,
        decision=DecisionKind.ALLOW,
        policy_id="p0",
        cost_model_id="flat",
    )


def _distorted_events(rng, n: int) -> list[DecisionEvent]:
    p = rng.uniform(0.02, 0.98, n)
    correct = rng.random(n) < p
    confidences = np.clip(np.sqrt(p), 0.001, 0.999)
    return [
        make_event(f"e-{i}", float(conf), bool(ok))
        for i, (conf, ok) in enumerate(zip(confidences, correct, strict=True))
    ]


def test_pipeline_improves_ece():
    rng = np.random.default_rng(21)
    events = _distorted_events(rng, 1500)
    result = run_calibration_pipeline(events, _cost_profile())
    assert result.n_labeled == 1500
    assert result.report["raw_ece"] > 0.05
    assert result.report["calibrated_ece"] < result.report["raw_ece"]


def test_unlabeled_events_retained():
    rng = np.random.default_rng(3)
    labeled = [
        make_event(f"l-{i}", float(rng.uniform(0.2, 0.9)), bool(rng.random() < 0.3))
        for i in range(100)
    ]
    unlabeled = [
        make_event(f"u-{i}", float(rng.uniform(0.2, 0.9)), None) for i in range(50)
    ]
    events = labeled + unlabeled
    result = run_calibration_pipeline(events, _cost_profile())
    assert result.n_labeled == 100
    assert result.n_unlabeled == 50
    assert len(result.calibrated) == 150
    for i, event in enumerate(events[100:], start=100):
        assert result.calibrated[i] == event.calibrated_probability


def test_label_fn_overrides():
    rng = np.random.default_rng(8)
    events = [
        make_event(f"e-{i}", float(rng.uniform(0.2, 0.9)), None) for i in range(200)
    ]
    result = run_calibration_pipeline(
        events,
        _cost_profile(),
        label_fn=lambda event: int(event.event_id.split("-")[1]) % 2 == 0,
    )
    assert result.n_labeled == 100
    assert result.n_unlabeled == 100


def test_threshold_fitted_in_range():
    rng = np.random.default_rng(4)
    events = _distorted_events(rng, 200)
    result = run_calibration_pipeline(events, _cost_profile())
    assert 0.0 <= result.threshold <= 1.0


def test_fit_policy_from_ledger():
    rng = np.random.default_rng(9)
    ledger = AuditLedger()
    for i in range(200):
        confidence = float(rng.uniform(0.2, 0.95))
        error = bool(rng.random() < 0.3)
        severity = "HIGH" if i % 3 == 0 else None
        ledger.append(make_event(f"e-{i}", confidence, error, severity=severity))
    bundle = fit_policy_from_ledger(ledger, _cost_profile(), policy_id="p-fitted")
    assert bundle.id == "p-fitted"
    assert bundle.cost_model_id == "flat"
    assert isinstance(bundle.escalation_threshold, float)
    assert bundle.revision.startswith("fitted-")
    assert 0.0 <= bundle.escalation_threshold <= 1.0
    ledger.close()


def test_deterministic():
    rng = np.random.default_rng(12)
    events = _distorted_events(rng, 300)
    profile = _cost_profile()
    first = run_calibration_pipeline(events, profile)
    second = run_calibration_pipeline(events, profile)
    assert first == second


def test_empty_events_safe():
    profile = _cost_profile()
    result = run_calibration_pipeline([], profile)
    assert result.n_labeled == 0
    assert result.n_unlabeled == 0
    assert result.calibrated == []
    assert result.threshold == 0.0
    assert result.report["n"] == 0
    assert result.report["calibrated_ece"] == 0.0
    bundle = fit_policy_from_ledger(AuditLedger(), profile, "p-empty")
    assert bundle.escalation_threshold == 0.0


def test_pipeline_uses_only_labeled_for_fit():
    # Unlabeled events must not influence the fitted threshold: running the
    # pipeline on (labeled + unlabeled) must yield the same threshold as
    # running it on the labeled subset alone.
    rng = np.random.default_rng(17)
    n = 300
    p = rng.uniform(0.02, 0.98, n)
    correct = rng.random(n) < p
    confidences = np.clip(np.sqrt(p), 0.001, 0.999)
    events = [
        make_event(f"e-{i}", float(conf), bool(ok) if i % 4 != 0 else None)
        for i, (conf, ok) in enumerate(zip(confidences, correct, strict=True))
    ]
    profile = _cost_profile()
    labeled = [
        e for e in events if (e.observed_outcome or {}).get("error") is not None
    ]
    full_result = run_calibration_pipeline(events, profile)
    labeled_result = run_calibration_pipeline(labeled, profile)
    assert full_result.threshold == pytest.approx(labeled_result.threshold, abs=1e-9)
    assert full_result.n_labeled == labeled_result.n_labeled
    assert full_result.n_unlabeled == n - len(labeled)
