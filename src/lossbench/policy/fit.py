"""Threshold and model-tier fitting from labeled decision history."""

from __future__ import annotations

from collections.abc import Sequence

from lossbench.metrics.coverage import risk_coverage_curve
from lossbench.metrics.loss import severity_weighted_loss
from lossbench.schema import CostProfile, Severity


def fit_escalation_threshold(
    probabilities: Sequence[float],
    errors: Sequence[bool],
    severities: Sequence[Severity],
    cost_profile: CostProfile,
    grid: int = 40,
) -> dict:
    """Grid-search tau in [0, 1] minimizing total severity-weighted policy cost.

    Cost per tau is the severity-weighted loss of unreviewed errors
    (risk_coverage_curve) plus escalate_cost per escalated case. Returns
    {"best_threshold": float, "best_cost": float, "baseline_cost": float,
    "n": int} where baseline_cost is the never-escalate policy's loss.
    """
    if not (len(probabilities) == len(errors) == len(severities)):
        raise ValueError("probabilities, errors, and severities must have equal length")
    n = len(probabilities)
    if n == 0:
        raise ValueError("at least one labeled decision is required")
    curve = risk_coverage_curve(probabilities, errors, severities, cost_profile, n_points=grid)

    def total_cost(point: dict) -> float:
        return point["loss"] + cost_profile.escalate_cost * point["escalated"]

    best = min(curve, key=total_cost)
    return {
        "best_threshold": best["threshold"],
        "best_cost": total_cost(best),
        "baseline_cost": severity_weighted_loss(errors, severities, cost_profile),
        "n": n,
    }


def fit_model_tiers(
    p_error_by_model: dict[str, float],
    severity: Severity,
    cost_profile: CostProfile,
    base_cost: dict[str, float],
) -> dict[str, float]:
    """Calibrate model_tiers so bayes_route routing cost scales with model risk.

    Rule: tier(m) = max(0, p_error[m] * K(severity) * 0.5), the simplest
    consistent rule from the spec: a model's routing tier is half its expected
    error cost at the calibrated p_error, so routing is decided by risk rather
    than price. `base_cost` is accepted for the min-cost-advantage variant and
    is not used by this rule.
    """
    k = cost_profile.cost(severity)
    return {model: max(0.0, p_error * k * 0.5) for model, p_error in p_error_by_model.items()}
