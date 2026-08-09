from datetime import UTC, datetime

import pytest

from lossbench.schema import CostProfile, DecisionEvent, DecisionKind, Severity
from lossbench.scoring.false_success import false_success_rate
from lossbench.scoring.passk import (
    outcome_verified_pass_at_k,
    pass_k_reliability,
    severity_corrected_passk_report,
)

ASYMMETRIC = CostProfile(
    id="asym-test",
    description="d",
    severity_costs={"LOW": 0.2, "MEDIUM": 1.0, "HIGH": 10.0, "CRITICAL": 50.0},
)


def _event(decision, observed_outcome=None, proposed_action=None):
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
        observed_outcome=observed_outcome,
        proposed_action=proposed_action,
        policy_id="pol",
        cost_model_id="cm",
    )


def _checker(events, gold):
    last = events[-1]
    return bool(last.proposed_action and last.proposed_action.get("tool") == gold.get("tool"))


def test_pass_at_k_values():
    trials = [[False, True, False], [False, False, True]]
    assert outcome_verified_pass_at_k(trials, 1) == 0.0
    assert outcome_verified_pass_at_k(trials, 2) == 0.5
    assert outcome_verified_pass_at_k(trials, 3) == 1.0
    assert outcome_verified_pass_at_k(trials, 0) == 0.0


def test_pass_k_reliability():
    trials = [[True, True, True], [True, False, True], [True, True]]
    assert pass_k_reliability(trials, 3) == 1 / 3
    assert outcome_verified_pass_at_k(trials, 3) == 1.0
    assert pass_k_reliability(trials, 3) <= outcome_verified_pass_at_k(trials, 3)
    all_success = [[True, True, True]]
    assert pass_k_reliability(all_success, 3) == 1.0
    assert outcome_verified_pass_at_k(all_success, 3) == 1.0


def test_false_success_detects_claim_without_effect():
    claim_false = [_event(DecisionKind.ALLOW, proposed_action={"tool": "apply"})]
    claim_true = [_event(DecisionKind.ALLOW, proposed_action={"tool": "commit"})]
    outcome_observed = [
        _event(
            DecisionKind.VERIFY,
            observed_outcome={"status": "ok"},
            proposed_action={"tool": "commit"},
        )
    ]
    denied = [_event(DecisionKind.DENY)]
    trajectories = [claim_false, claim_true, outcome_observed, denied]
    golds = [{"tool": "commit"}] * 4
    assert false_success_rate(trajectories, golds, _checker) == 0.25
    assert false_success_rate([claim_false], [{"tool": "commit"}], _checker) == 1.0
    assert false_success_rate([claim_true], [{"tool": "commit"}], _checker) == 0.0


def test_false_success_rate_range():
    claim_false = [_event(DecisionKind.ALLOW, proposed_action={"tool": "apply"})]
    claim_true = [_event(DecisionKind.ALLOW, proposed_action={"tool": "commit"})]
    denied = [_event(DecisionKind.DENY)]
    trajectories = [claim_false, claim_true, denied, claim_false]
    golds = [{"tool": "commit"}] * 4
    rate = false_success_rate(trajectories, golds, _checker)
    assert 0.0 <= rate <= 1.0
    all_false = [claim_false, claim_false, claim_false]
    assert false_success_rate(all_false, [{"tool": "commit"}] * 3, _checker) == 1.0


def test_severity_corrected_penalizes_high_failures():
    severities = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]
    fails_high = [[True], [True], [False]]
    fails_low = [[False], [True], [True]]
    rep_high = severity_corrected_passk_report(fails_high, severities, ASYMMETRIC, 1)
    rep_low = severity_corrected_passk_report(fails_low, severities, ASYMMETRIC, 1)
    assert rep_high["pass@k"] == pytest.approx(2 / 3)
    assert rep_low["pass@k"] == pytest.approx(2 / 3)
    assert rep_high["pass^k"] == rep_low["pass^k"]
    assert rep_high["severity_weighted_passk"] < rep_low["severity_weighted_passk"]
    assert rep_high["severity_weighted_passk"] == pytest.approx(1.2 / 11.2)
    assert rep_low["severity_weighted_passk"] == pytest.approx(11.0 / 11.2)


def test_empty_inputs_safe():
    assert outcome_verified_pass_at_k([], 3) == 0.0
    assert pass_k_reliability([], 3) == 0.0
    assert false_success_rate([], [], _checker) == 0.0
    report = severity_corrected_passk_report([], [], ASYMMETRIC, 3)
    assert report == {"pass@k": 0.0, "pass^k": 0.0, "severity_weighted_passk": 0.0}


def test_deterministic():
    trials = [[False, True, False], [False, False, True]]
    severities = [Severity.LOW, Severity.HIGH]
    trajectories = [
        [_event(DecisionKind.ALLOW, proposed_action={"tool": "apply"})],
        [_event(DecisionKind.ALLOW, proposed_action={"tool": "commit"})],
    ]
    golds = [{"tool": "commit"}] * 2
    first = (
        outcome_verified_pass_at_k(trials, 2),
        pass_k_reliability(trials, 2),
        false_success_rate(trajectories, golds, _checker),
        severity_corrected_passk_report(trials, severities, ASYMMETRIC, 2),
    )
    second = (
        outcome_verified_pass_at_k(trials, 2),
        pass_k_reliability(trials, 2),
        false_success_rate(trajectories, golds, _checker),
        severity_corrected_passk_report(trials, severities, ASYMMETRIC, 2),
    )
    assert first == second
