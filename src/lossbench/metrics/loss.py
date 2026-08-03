"""Pure loss math: the formal model from the design spec (section 5).

All functions are pure; no I/O. Costs always come from a CostProfile
(never a hidden constant). Regret is relative to an explicit baseline.
"""

from __future__ import annotations

from collections.abc import Sequence

from lossbench.schema import CostProfile, Severity

DEFAULT_K_FLAT = 1.0


def expected_decision_cost(p: float, severity: Severity, profile: CostProfile) -> float:
    """E[loss] of a single auto-decision = P(error) * K(severity)."""
    return p * profile.cost(severity)


def severity_weighted_loss(
    errors: Sequence[bool], severities: Sequence[Severity], profile: CostProfile
) -> float:
    """Realized severity-weighted loss over a batch: sum of K(sigma) for errors."""
    if len(errors) != len(severities):
        raise ValueError("errors and severities must have equal length")
    return sum(profile.cost(sev) for err, sev in zip(errors, severities, strict=True) if err)


def total_policy_loss(
    errors: Sequence[bool],
    severities: Sequence[Severity],
    profile: CostProfile,
    model_cost: float = 0.0,
    judge_cost: float = 0.0,
    human_cost: float = 0.0,
) -> float:
    """Total realized policy loss: business error loss plus execution costs.

    Judge/human costs must be passed only when actually incurred (conditional
    invocation). Unconditional costs are common to all policies and cancel in
    comparisons; do not subtract them per-decision.
    """
    return (
        severity_weighted_loss(errors, severities, profile)
        + model_cost
        + judge_cost
        + human_cost
    )


def regret(realized: float, baseline: float) -> float:
    """Regret vs an explicit baseline policy (oracle, always-cheap, etc.)."""
    return realized - baseline


def loss_at_fixed_budget(curve: Sequence[dict[str, float]], budget: float) -> float:
    """Best achievable loss along a risk-coverage curve at or under `budget`.

    `curve` items must contain the keys 'review_load' and 'loss'.
    """
    feasible = [point["loss"] for point in curve if point["review_load"] <= budget]
    if not feasible:
        raise ValueError("budget below the minimum point on the curve")
    return min(feasible)
