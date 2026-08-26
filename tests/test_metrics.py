import pytest

from lossbench.metrics.calibration import brier_score, ece, reliability_curve
from lossbench.metrics.coverage import risk_coverage_curve
from lossbench.metrics.deferral import ask_f1, escalation_precision_recall, missed_high_loss_rate
from lossbench.metrics.loss import (
    expected_decision_cost,
    loss_at_fixed_budget,
    regret,
    severity_weighted_loss,
    total_policy_loss,
)
from lossbench.schema import CostProfile, Severity

FLAT = CostProfile(
    id="flat-test",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 1.0},
)

ASYMMETRIC = CostProfile(
    id="asym-test",
    description="d",
    severity_costs={"LOW": 0.2, "MEDIUM": 1.0, "HIGH": 10.0, "CRITICAL": 50.0},
)


def test_severity_weighted_loss_counts_only_errors():
    errors = [True, False, True, False]
    sevs = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    assert severity_weighted_loss(errors, sevs, ASYMMETRIC) == pytest.approx(10.2)


def test_severity_weighted_loss_length_mismatch():
    with pytest.raises(ValueError):
        severity_weighted_loss([True], [], ASYMMETRIC)


def test_total_policy_loss_adds_execution_costs():
    errors = [True]
    sevs = [Severity.HIGH]
    total = total_policy_loss(errors, sevs, ASYMMETRIC, model_cost=1.0, judge_cost=0.5)
    assert total == pytest.approx(11.5)


def test_regret_sign():
    assert regret(5.0, 3.0) == 2.0
    assert regret(2.0, 5.0) == -3.0


def test_expected_decision_cost():
    assert expected_decision_cost(0.5, Severity.HIGH, ASYMMETRIC) == pytest.approx(5.0)


def test_risk_coverage_curve_monotonic_load():
    probs = [0.9, 0.8, 0.5, 0.2, 0.1]
    errors = [True, True, False, True, False]
    sevs = [Severity.HIGH] * 5
    curve = risk_coverage_curve(probs, errors, sevs, ASYMMETRIC, n_points=10)
    assert len(curve) == 11
    loads = [p["review_load"] for p in curve]
    assert loads == sorted(loads, reverse=True)
    # at tau=0 everything is escalated: no business loss remains
    assert curve[0]["loss"] == 0.0
    # at tau=1 nothing is escalated: loss = full realized severity loss (3 errors x 10)
    assert curve[-1]["loss"] == pytest.approx(30.0)


def test_risk_coverage_curve_handles_empty():
    assert risk_coverage_curve([], [], [], ASYMMETRIC) == []


def test_loss_at_fixed_budget():
    curve = [
        {"review_load": 0.0, "loss": 100.0},
        {"review_load": 0.5, "loss": 40.0},
        {"review_load": 1.0, "loss": 0.0},
    ]
    assert loss_at_fixed_budget(curve, budget=0.6) == pytest.approx(40.0)


def test_loss_at_fixed_budget_below_minimum():
    with pytest.raises(ValueError):
        loss_at_fixed_budget([{"review_load": 0.8, "loss": 10.0}], budget=0.5)


def test_ece_perfectly_calibrated():
    confs = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    result = ece(confs, correct, n_bins=10)
    assert result["ece"] == pytest.approx(0.0, abs=1e-6)


def test_ece_known_value():
    # conf 0.9, wrong -> |acc-conf| = 0.9 in its bin
    confs = [0.9]
    correct = [False]
    assert ece(confs, correct, n_bins=10)["ece"] == pytest.approx(0.9)


def test_reliability_curve_bins():
    curve = reliability_curve([0.95, 0.85, 0.05], [True, True, False], n_bins=10)
    assert all("conf_mean" in p and "accuracy" in p and "count" in p for p in curve)


def test_brier_score():
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier_score([0.5], [True]) == pytest.approx(0.25)


def test_escalation_precision_recall():
    escalated = [True, True, False, False]
    should = [True, False, True, False]
    result = escalation_precision_recall(escalated, should)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_ask_f1_penalizes_spam():
    spam = ask_f1(question_precision=[0.1, 0.1], blocker_recall=[1.0, 1.0])
    sharp = ask_f1(question_precision=[1.0, 1.0], blocker_recall=[1.0, 1.0])
    assert spam["ask_f1"] < sharp["ask_f1"]
    assert sharp["ask_f1"] == pytest.approx(1.0)


def test_missed_high_loss_rate():
    errors = [True, True, False]
    sevs = [Severity.HIGH, Severity.CRITICAL, Severity.HIGH]
    escalated = [False, True, False]
    # denominator = error weight only: 10 + 50 = 60; missed = unescalated HIGH = 10
    assert missed_high_loss_rate(errors, sevs, ASYMMETRIC, escalated) == pytest.approx(
        10.0 / 60.0, abs=1e-4
    )


def test_missed_high_loss_rate_no_high_weight():
    # No HIGH/CRITICAL error weight exists (only a LOW-severity error) — nothing
    # was missed, so this is the best case (0.0), not the worst (1.0).
    assert (
        missed_high_loss_rate(
            [True], [Severity.LOW], ASYMMETRIC, escalated=[False]
        )
        == 0.0
    )
