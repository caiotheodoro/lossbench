import pytest

from lossbench.decision import bayes_route, escalate_iff, expected_escalation_gain
from lossbench.schema import CostProfile, Severity

ASYMMETRIC = CostProfile(
    id="asym-test",
    description="d",
    severity_costs={"LOW": 0.2, "MEDIUM": 1.0, "HIGH": 10.0, "CRITICAL": 50.0},
    escalate_cost=1.0,
)

REVIEW_HEAVY = CostProfile(
    id="review-heavy-test",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 5.0, "HIGH": 50.0, "CRITICAL": 250.0},
    escalate_cost=25.0,
)


def test_expected_escalation_gain():
    gain = expected_escalation_gain(0.5, Severity.HIGH, ASYMMETRIC, judge_cost=0.1)
    assert gain == pytest.approx(4.9)


def test_escalate_iff_threshold_behavior():
    # gain > escalate_cost => escalate
    assert escalate_iff(0.5, Severity.HIGH, ASYMMETRIC) is True
    assert escalate_iff(0.05, Severity.LOW, ASYMMETRIC) is False


def test_escalate_iff_includes_judge_cost_when_conditional():
    # judge cost flips a borderline HIGH decision
    assert escalate_iff(0.11, Severity.HIGH, ASYMMETRIC, judge_cost=0.2) is False
    assert escalate_iff(0.11, Severity.HIGH, ASYMMETRIC, judge_cost=0.0) is True


def test_escalate_iff_review_heavy_environment():
    # expensive reviewers: only large exposure escalates
    assert escalate_iff(0.9, Severity.HIGH, REVIEW_HEAVY) is True
    assert escalate_iff(0.4, Severity.HIGH, REVIEW_HEAVY) is False


def test_bayes_route_picks_min_expected_cost():
    p_error = {"cheap": 0.3, "frontier": 0.05}
    model_cost = {"cheap": 0.01, "frontier": 1.0}
    # cheap: 0.3*10 + 0.01 = 3.01 ; frontier: 0.05*10 + 1.0 = 1.5
    best, expected = bayes_route(p_error, Severity.HIGH, ASYMMETRIC, model_cost)
    assert best == "frontier"
    assert expected == pytest.approx(1.5)


def test_bayes_route_cheap_wins_when_costs_flat_low():
    p_error = {"cheap": 0.3, "frontier": 0.05}
    model_cost = {"cheap": 0.01, "frontier": 1.0}
    # LOW: cheap = 0.3*0.2+0.01 = 0.07 ; frontier = 0.05*0.2+1.0 = 1.01
    best, _ = bayes_route(p_error, Severity.LOW, ASYMMETRIC, model_cost)
    assert best == "cheap"


def test_bayes_route_requires_same_keys():
    with pytest.raises(ValueError):
        bayes_route({"a": 0.1}, Severity.HIGH, ASYMMETRIC, {"a": 1.0, "b": 2.0})


def test_bayes_route_requires_candidates():
    with pytest.raises(ValueError):
        bayes_route({}, Severity.HIGH, ASYMMETRIC, {})
