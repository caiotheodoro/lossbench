"""Deferral / escalation metrics.

Escalation quality matters beyond raw rate: escalation spam (asking for help
on everything) must be penalized. Ask-F1-style metrics follow HiL-Bench's
architecture of being immune to question-spam gaming.
"""

from __future__ import annotations

from collections.abc import Sequence

from lossbench.schema import CostProfile, Severity


def escalation_precision_recall(
    escalated: Sequence[bool], should_escalate: Sequence[bool]
) -> dict[str, float]:
    """Precision and recall of the escalation decision.

    should_escalate = ground truth: the case genuinely needed human review.
    """
    if len(escalated) != len(should_escalate):
        raise ValueError("escalated and should_escalate must have equal length")
    n = len(escalated)
    if n == 0:
        return {"precision": 1.0, "recall": 1.0, "n": 0}
    tp = sum(1 for e, s in zip(escalated, should_escalate, strict=True) if e and s)
    fp = sum(1 for e, s in zip(escalated, should_escalate, strict=True) if e and not s)
    fn = sum(1 for e, s in zip(escalated, should_escalate, strict=True) if not e and s)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "n": n}


def ask_f1(
    question_precision: Sequence[float], blocker_recall: Sequence[float]
) -> dict[str, float]:
    """F1 over escalation quality scores per case.

    question_precision: how targeted each question/escalation is (0..1).
    blocker_recall: whether the real blocker was identified (0..1).
    Returns the harmonic mean aggregated over cases; spam questions drag
    precision down per case, so high volume of low-quality asks cannot win.
    """
    if len(question_precision) != len(blocker_recall):
        raise ValueError("question_precision and blocker_recall must have equal length")
    if not question_precision:
        return {"ask_f1": 1.0, "n": 0}
    scores = [
        (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        for p, r in zip(question_precision, blocker_recall, strict=True)
    ]
    return {"ask_f1": round(sum(scores) / len(scores), 4), "n": len(scores)}


def missed_high_loss_rate(
    errors: Sequence[bool],
    severities: Sequence[Severity],
    profile: CostProfile,
    escalated: Sequence[bool],
) -> float:
    """Share of high-severity ERROR weight that was not caught by escalation.

    An error at severity HIGH or CRITICAL that was neither escalated nor caught
    counts as missed high-severity loss. The denominator is the weight of
    ERRORS only (successful cases never carried loss, so they must not dilute
    the ratio). Returns 1.0 when no high-severity error weight exists,
    otherwise missed_weight / total_high_error_weight.
    """
    if not (len(errors) == len(severities) == len(escalated)):
        raise ValueError("errors, severities, escalated must have equal length")
    total_high = 0.0
    missed = 0.0
    for err, sev, esc in zip(errors, severities, escalated, strict=True):
        if err and sev in (Severity.HIGH, Severity.CRITICAL):
            k = profile.cost(sev)
            total_high += k
            if not esc:
                missed += k
    if total_high == 0.0:
        return 1.0
    return round(missed / total_high, 4)
