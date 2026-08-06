from datetime import datetime

from lossbench.features import (
    RISK_FEATURES,
    extract_risk_features,
    extract_trajectory_features,
    feature_matrix,
)
from lossbench.schema import DecisionEvent, DecisionKind


def _event(**overrides) -> DecisionEvent:
    base = dict(
        event_id="e1",
        trace_id="t1",
        trajectory_id="tr1",
        task_id="task1",
        timestamp=datetime(2026, 1, 1),
        input_snapshot_hash="h1",
        prompt_hash="p1",
        model_id="m1",
        decision=DecisionKind.ALLOW,
        policy_id="pol1",
        cost_model_id="c1",
    )
    base.update(overrides)
    return DecisionEvent(**base)


def test_per_event_keys_exact():
    event = _event(
        calibrated_probability=0.87,
        latency_ms=123.4,
        token_usage={"prompt_tokens": 42, "agreement": 2},
        risk_features={"tool_calls_so_far": 3.0, "trajectory_len": 5.0},
        observed_outcome={"verifier_disagreed": True},
    )
    feats = extract_risk_features(event)
    assert set(feats) == set(RISK_FEATURES)
    assert feats["confidence"] == 0.87
    assert feats["input_length_tokens"] == 42.0
    assert feats["latency_ms"] == 123.4
    assert feats["tool_calls_so_far"] == 3.0
    assert feats["agreement_across_samples"] == 2.0
    assert feats["verifier_disagreement"] == 1.0
    assert feats["trajectory_len"] == 5.0


def test_missing_fields_default():
    feats = extract_risk_features(_event())
    assert set(feats) == set(RISK_FEATURES)
    assert feats["confidence"] == 0.0
    assert feats["verifier_disagreement"] == 0.0
    assert feats["agreement_across_samples"] == 1.0
    assert feats["input_length_tokens"] == 0.0
    assert feats["tool_calls_so_far"] == 0.0


def test_escalation_count():
    events = [
        _event(event_id=f"e{i}", decision=kind)
        for i, kind in enumerate(
            [DecisionKind.ESCALATE, DecisionKind.ESCALATE, DecisionKind.ALLOW]
        )
    ]
    agg = extract_trajectory_features(events)
    assert agg["escalation_count"] == 2.0
    assert agg["abstain_count"] == 0.0
    assert agg["deny_count"] == 0.0


def test_total_cost_sum():
    events = [
        _event(event_id=f"e{i}", model_cost=float(cost))
        for i, cost in enumerate([1, 2, 3])
    ]
    agg = extract_trajectory_features(events)
    assert agg["total_cost"] == 6.0
    assert agg["total_latency_ms"] == 0.0
    assert agg["confidence_mean"] == 0.0


def test_feature_matrix_alignment():
    events = [_event(event_id=f"e{i}") for i in range(3)]
    columns, rows = feature_matrix(events)
    assert len(rows) == len(events)
    assert len(columns) == len(RISK_FEATURES) + len(extract_trajectory_features(events))
    for row in rows:
        assert len(row) == len(columns)


def test_deterministic():
    events = [
        _event(
            event_id=f"e{i}",
            decision=DecisionKind.ESCALATE if i % 2 else DecisionKind.ALLOW,
            model_cost=float(i),
            latency_ms=float(i * 10),
        )
        for i in range(4)
    ]
    assert extract_risk_features(events[0]) == extract_risk_features(events[0])
    assert extract_trajectory_features(events) == extract_trajectory_features(events)
    assert feature_matrix(events) == feature_matrix(events)


def test_empty_trajectory_safe():
    agg = extract_trajectory_features([])
    assert set(agg) == {"confidence_mean", "confidence_max", "confidence_std",
                        "escalation_count", "abstain_count", "deny_count",
                        "total_cost", "total_latency_ms", "mean_input_tokens"}
    assert all(value == 0.0 for value in agg.values())
    columns, rows = feature_matrix([])
    assert rows == []
    assert len(columns) == len(RISK_FEATURES) + len(agg)
