import math
from datetime import UTC, datetime

import pytest

from lossbench.schema import DecisionEvent, DecisionKind
from lossbench.scoring import (
    ece_over_trajectories,
    tps_report,
    trajectory_proper_score,
    trajectory_success_probs,
)


def make_event(decision=DecisionKind.ALLOW, p=None) -> DecisionEvent:
    return DecisionEvent(
        event_id="e",
        trace_id="t",
        trajectory_id="tr",
        task_id="task",
        timestamp=datetime.now(UTC),
        input_snapshot_hash="i",
        prompt_hash="p",
        model_id="m",
        decision=decision,
        policy_id="pol",
        cost_model_id="cm",
        calibrated_probability=p,
    )


def test_empty_trajectory_zero():
    assert trajectory_proper_score([], True) == 0.0
    assert trajectory_proper_score([], False) == 0.0
    report = tps_report([])
    assert report == {
        "n": 0,
        "mean_tps": 0.0,
        "std_tps": 0.0,
        "median_tps": 0.0,
        "worst": 0.0,
        "best": 0.0,
    }


def test_perfect_predictions_score_zero():
    events = [make_event(p=0.0) for _ in range(3)]
    assert trajectory_proper_score(events, True) == pytest.approx(0.0, abs=1e-3)


def test_confident_wrong_penalized():
    perfect = trajectory_proper_score([make_event(p=0.0)], True)
    wrong = trajectory_proper_score([make_event(p=0.05)], False)
    assert wrong > perfect
    assert wrong == pytest.approx(0.95**2)


def test_deterministic():
    events = [make_event(p=0.3), make_event(p=0.7)]
    assert trajectory_proper_score(events, False) == trajectory_proper_score(
        events, False
    )
    assert trajectory_success_probs(events) == trajectory_success_probs(events)


def test_report_statistics():
    trajectories = [
        ([make_event(p=0.2)], True),
        ([make_event(p=0.6)], True),
        ([make_event(p=0.1)], False),
    ]
    scores = [0.04, 0.36, 0.81]
    report = tps_report(trajectories)
    assert report["n"] == 3
    assert report["mean_tps"] == pytest.approx(sum(scores) / 3)
    assert report["median_tps"] == pytest.approx(0.36)
    assert report["worst"] == pytest.approx(0.81)
    assert report["best"] == pytest.approx(0.04)
    mean = sum(scores) / 3
    expected_std = math.sqrt(sum((s - mean) ** 2 for s in scores) / 3)
    assert report["std_tps"] == pytest.approx(expected_std)


def test_success_probs_length():
    events = [
        make_event(p=0.2),
        make_event(decision=DecisionKind.DENY, p=0.8),
        make_event(p=0.0),
    ]
    probs = trajectory_success_probs(events)
    assert len(probs) == len(events)
    assert all(0.01 <= q <= 0.99 for q in probs)
    assert trajectory_success_probs([]) == []


def test_none_probability_treated_as_half():
    events = [make_event(p=None)]
    assert trajectory_success_probs(events) == [0.5]
    assert trajectory_proper_score(events, True) == pytest.approx(0.25)
    assert trajectory_proper_score(events, False) == pytest.approx(0.25)


def test_ece_over_trajectories_valid_range():
    trajs = [
        ([make_event(p=0.1)], True),
        ([make_event(p=0.9)], False),
        ([make_event(p=0.5)], True),
        ([make_event(p=0.3)], False),
    ]
    result = ece_over_trajectories(
        [t for t, _ in trajs], [s for _, s in trajs]
    )
    assert 0.0 <= result <= 1.0
    assert ece_over_trajectories([], []) == 0.0
