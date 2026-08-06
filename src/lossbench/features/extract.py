"""Pure, deterministic feature extraction from DecisionEvent records."""

from __future__ import annotations

import math
from collections.abc import Sequence

from lossbench.schema import DecisionEvent, DecisionKind

RISK_FEATURES = (
    "confidence",
    "input_length_tokens",
    "latency_ms",
    "tool_calls_so_far",
    "agreement_across_samples",
    "verifier_disagreement",
    "trajectory_len",
)

_AGGREGATE_KEYS = (
    "confidence_mean",
    "confidence_max",
    "confidence_std",
    "escalation_count",
    "abstain_count",
    "deny_count",
    "total_cost",
    "total_latency_ms",
    "mean_input_tokens",
)


def extract_risk_features(event: DecisionEvent) -> dict[str, float]:
    """Feature vector for one decision event. Keys exactly as in RISK_FEATURES.
    confidence = calibrated_probability (0.0 when None);
    verifier_disagreement = 1.0 when the observed_outcome contains a
    "verifier_disagreed": true marker else 0.0; agreement_across_samples comes
    from token_usage key "agreement" if present else 1.0; input_length_tokens
    from token_usage "prompt_tokens" (0 when absent)."""
    outcome = event.observed_outcome or {}
    return {
        "confidence": event.calibrated_probability or 0.0,
        "input_length_tokens": float(event.token_usage.get("prompt_tokens", 0)),
        "latency_ms": float(event.latency_ms),
        "tool_calls_so_far": float(event.risk_features.get("tool_calls_so_far", 0.0)),
        "agreement_across_samples": float(event.token_usage.get("agreement", 1)),
        "verifier_disagreement": 1.0 if outcome.get("verifier_disagreed") else 0.0,
        "trajectory_len": float(event.risk_features.get("trajectory_len", 0.0)),
    }


def extract_trajectory_features(events: Sequence[DecisionEvent]) -> dict[str, float]:
    """Aggregates over a trajectory: mean/max/std of per-event confidence,
    count of ESCALATE/ABSTAIN/DENY decisions, total model cost, total
    latency, mean input_length_tokens."""
    per_event = [extract_risk_features(event) for event in events]
    confidences = [feats["confidence"] for feats in per_event]
    input_lengths = [feats["input_length_tokens"] for feats in per_event]
    n = len(events)
    return {
        "confidence_mean": sum(confidences) / n if n else 0.0,
        "confidence_max": max(confidences) if n else 0.0,
        "confidence_std": _std(confidences),
        "escalation_count": float(
            sum(1 for event in events if event.decision == DecisionKind.ESCALATE)
        ),
        "abstain_count": float(
            sum(1 for event in events if event.decision == DecisionKind.ABSTAIN)
        ),
        "deny_count": float(
            sum(1 for event in events if event.decision == DecisionKind.DENY)
        ),
        "total_cost": float(sum(event.model_cost for event in events)),
        "total_latency_ms": float(sum(event.latency_ms for event in events)),
        "mean_input_tokens": sum(input_lengths) / n if n else 0.0,
    }


def feature_matrix(events: Sequence[DecisionEvent]) -> tuple[list[str], list[list[float]]]:
    """Returns (columns, rows) where columns = RISK_FEATURES + aggregate keys
    merged; rows aligned per event. Used by the calibrate pipeline."""
    aggregate = extract_trajectory_features(events)
    columns = list(RISK_FEATURES) + list(_AGGREGATE_KEYS)
    rows = [
        [feats[key] for key in RISK_FEATURES] + [aggregate[key] for key in _AGGREGATE_KEYS]
        for feats in (extract_risk_features(event) for event in events)
    ]
    return columns, rows


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
