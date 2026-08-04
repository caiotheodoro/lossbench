"""Bayes-optimal decision core (design spec section 5).

Pure functions translating calibrated risk + costs into routing and
escalation decisions. Judge cost must be included only when conditionally
invoked; unconditional judge cost is common to all policies and cancels.
"""

from __future__ import annotations

from lossbench.schema import CostProfile, Severity


def expected_escalation_gain(
    p_avoidable_error: float,
    severity: Severity,
    profile: CostProfile,
    judge_cost: float = 0.0,
) -> float:
    """Expected avoided loss from reviewing a case now.

    gain = P(avoidable error) * K(severity) - judge_cost
    Human review cost is NOT included here; see escalate_iff.
    """
    return p_avoidable_error * profile.cost(severity) - judge_cost


def escalate_iff(
    p_avoidable_error: float,
    severity: Severity,
    profile: CostProfile,
    judge_cost: float = 0.0,
) -> bool:
    """Escalate iff expected avoided loss exceeds incremental review cost.

    Incremental cost = judge (if conditionally invoked) + human review.
    """
    gain = expected_escalation_gain(p_avoidable_error, severity, profile, judge_cost)
    return gain > profile.escalate_cost


def bayes_route(
    p_error: dict[str, float],
    severity: Severity,
    profile: CostProfile,
    model_cost: dict[str, float],
) -> tuple[str, float]:
    """Pick the model minimizing E[loss] = P(error|m)*K(severity) + price(m).

    Returns (best_model, expected_cost). Requires identical key sets.
    """
    if set(p_error) != set(model_cost):
        raise ValueError("p_error and model_cost must share the same model keys")
    if not p_error:
        raise ValueError("at least one model candidate is required")
    best_model = min(
        p_error,
        key=lambda m: p_error[m] * profile.cost(severity) + model_cost[m],
    )
    expected = p_error[best_model] * profile.cost(severity) + model_cost[best_model]
    return best_model, expected
