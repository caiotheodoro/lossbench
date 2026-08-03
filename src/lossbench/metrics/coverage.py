"""Risk-coverage analysis: expected loss vs escalation/review load."""

from __future__ import annotations

from collections.abc import Sequence

from lossbench.metrics.loss import severity_weighted_loss
from lossbench.schema import CostProfile, Severity


def risk_coverage_curve(
    probs: Sequence[float],
    errors: Sequence[bool],
    severities: Sequence[Severity],
    profile: CostProfile,
    n_points: int = 20,
) -> list[dict[str, float]]:
    """Expected loss vs review load as the escalation threshold tau varies.

    Escalation policy: escalate (review) any decision with calibrated p >= tau.
    Escalated cases are assumed reviewed and corrected (no business loss);
    unreviewed cases incur severity-weighted loss on error. Review load is the
    fraction of cases escalated. Each point: {threshold, loss, review_load,
    escalated}.
    """
    if not (len(probs) == len(errors) == len(severities)):
        raise ValueError("probs, errors, severities must have equal length")
    n = len(probs)
    if n == 0:
        return []
    curve: list[dict[str, float]] = []
    for i in range(n_points + 1):
        tau = i / n_points
        reviewed = [p >= tau for p in probs]
        unreviewed_errors = [
            err for err, rev in zip(errors, reviewed, strict=True) if not rev
        ]
        unreviewed_sevs = [
            sev for sev, rev in zip(severities, reviewed, strict=True) if not rev
        ]
        loss = severity_weighted_loss(unreviewed_errors, unreviewed_sevs, profile)
        curve.append(
            {
                "threshold": tau,
                "loss": loss,
                "review_load": sum(reviewed) / n,
                "escalated": float(sum(reviewed)),
            }
        )
    return curve
