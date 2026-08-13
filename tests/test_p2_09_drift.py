"""P2.9 drift monitor acceptance tests: PSI/KS drift, recalibration
triggers, fail-safe escalation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from lossbench.drift import DriftMonitor, ks_drift, psi, realized_ece
from lossbench.schema import DecisionEvent, DecisionKind


def _event(
    event_id: str,
    p: float,
    loss: float,
    error: bool | None = None,
) -> DecisionEvent:
    outcome: dict = {}
    if error is not None:
        outcome["error"] = error
    return DecisionEvent(
        event_id=event_id,
        tenant_id="default",
        trace_id="t",
        trajectory_id="tr",
        task_id="task",
        timestamp=datetime(2026, 8, 1),
        input_snapshot_hash="i",
        prompt_hash="p",
        model_id="m",
        calibrated_probability=p,
        expected_loss=loss,
        observed_outcome=outcome,
        decision=DecisionKind.ALLOW,
        policy_id="pol",
        cost_model_id="reconciliation",
    )


def _batch(ids: list[str], p: list[float], loss: list[float], error: list[bool | None]):
    return [
        _event(eid, pi, li, ei)
        for eid, pi, li, ei in zip(ids, p, loss, error, strict=True)
    ]


def test_psi_same_distribution_zero():
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, 2000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-12)


def test_psi_shifts_positive():
    rng = np.random.default_rng(7)
    baseline = rng.normal(1.0, 0.05, 2000)
    shifted = rng.normal(1.5, 0.05, 2000)
    assert psi(baseline, shifted) > 0.25


def test_clean_window_no_alerts():
    rng = np.random.default_rng(11)
    n = 300
    p = rng.uniform(0.2, 0.8, n).tolist()
    loss = rng.normal(1.0, 0.1, n).tolist()
    error = [pi > 0.5 for pi in p]
    baseline = _batch([f"b{i}" for i in range(n)], p, loss, error)
    window = _batch([f"w{i}" for i in range(n)], p, loss, error)
    monitor = DriftMonitor()
    monitor.fit_baseline(baseline)
    reports = monitor.detect(window)
    assert len(reports) == 3
    for report in reports:
        assert report.alert is False
        assert report.direction == "ok"
    assert monitor.escalation_override(reports) is False
    assert monitor.recalibration_needed(reports) is False


def test_loss_shift_triggers_fail_safe():
    rng = np.random.default_rng(5)
    n = 500
    p = rng.uniform(0.1, 0.6, n).tolist()
    baseline = _batch([f"b{i}" for i in range(n)], p, [1.0] * n, [False] * n)
    window = _batch([f"w{i}" for i in range(n)], p, [100.0] * n, [False] * n)
    monitor = DriftMonitor()
    monitor.fit_baseline(baseline)
    reports = monitor.detect(window)
    loss_report = next(r for r in reports if r.feature == "expected_loss_distribution")
    assert loss_report.alert is True
    assert loss_report.direction == "fail_safe_escalate"
    assert monitor.escalation_override(reports) is True


def test_p_shift_triggers_recalibrate():
    rng = np.random.default_rng(9)
    n = 500
    p_low = rng.uniform(0.05, 0.15, n).tolist()
    p_high = rng.uniform(0.8, 0.95, n).tolist()
    baseline = _batch([f"b{i}" for i in range(n)], p_low, [1.0] * n, [False] * n)
    window = _batch([f"w{i}" for i in range(n)], p_high, [1.0] * n, [True] * n)
    monitor = DriftMonitor()
    monitor.fit_baseline(baseline)
    reports = monitor.detect(window)
    p_report = next(r for r in reports if r.feature == "calibrated_p")
    assert p_report.alert is True
    assert p_report.direction == "recalibrate"
    assert monitor.recalibration_needed(reports) is True


def test_realized_ece_delta():
    baseline = _batch(
        [f"b{i}" for i in range(20)],
        [1.0 if i % 2 == 0 else 0.0 for i in range(20)],
        [1.0] * 20,
        [True if i % 2 == 0 else False for i in range(20)],
    )
    window = _batch(
        [f"w{i}" for i in range(10)],
        [0.9] * 10,
        [1.0] * 10,
        [True if i % 2 == 0 else False for i in range(10)],
    )
    baseline_ece = realized_ece(baseline)
    assert baseline_ece == pytest.approx(0.0)
    monitor = DriftMonitor()
    monitor.fit_baseline(baseline)
    reports = monitor.detect(window)
    ece_report = next(r for r in reports if r.feature == "realized_ece")
    assert realized_ece(window) - baseline_ece > monitor.ece_delta_threshold
    assert ece_report.alert is True
    assert ece_report.direction == "recalibrate"


def test_no_baseline_returns_empty():
    monitor = DriftMonitor()
    assert monitor.detect([]) == []


def test_empty_window_returns_empty():
    baseline = _batch(["b0"], [0.5], [1.0], [False])
    monitor = DriftMonitor()
    monitor.fit_baseline(baseline)
    assert monitor.detect([]) == []


def test_deterministic():
    rng = np.random.default_rng(3)
    n = 200
    p = rng.uniform(0.2, 0.8, n).tolist()
    loss = rng.normal(1.0, 0.1, n).tolist()
    error = [pi > 0.5 for pi in p]
    baseline = _batch([f"b{i}" for i in range(n)], p, loss, error)
    window = _batch([f"w{i}" for i in range(n)], p, loss, error)
    monitor = DriftMonitor()
    monitor.fit_baseline(baseline)
    assert monitor.detect(window) == monitor.detect(window)


def test_ks_drift_known():
    rng = np.random.default_rng(13)
    a = rng.uniform(0.0, 1.0, 500).tolist()
    statistic, p_value = ks_drift(a, a)
    assert p_value == pytest.approx(1.0)
    assert statistic == pytest.approx(0.0, abs=1e-12)
