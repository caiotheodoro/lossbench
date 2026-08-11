"""Replay lab: deterministic policy-only counterfactual simulation.

The flagship demo: "re-run last month's workload under a different risk
policy". Pure function of recorded DecisionEvents + a cost profile — NO LLM
calls, deterministic by construction. Escalating a case removes its business
loss (reviewed cases are corrected) at escalate_cost; unreviewed erroneous
cases pay K(severity).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from lossbench.ledger.store import AuditLedger
from lossbench.metrics.loss import severity_weighted_loss
from lossbench.schema import CostProfile, DecisionEvent, DecisionKind, PolicyBundle, Severity


@dataclass
class ReplayOutcome:
    """Counterfactual comparison of two escalation thresholds."""

    before_loss: float
    after_loss: float
    before_review_load: float
    after_review_load: float
    total_before: float
    total_after: float
    per_case_diff: list[dict] = field(default_factory=list)


def _severity(event: DecisionEvent) -> Severity:
    """Severity lives in observed_outcome (risk_features is float-typed)."""
    outcome = event.observed_outcome or {}
    raw = outcome.get("severity") or event.risk_features.get("severity")
    try:
        return Severity(raw) if raw else Severity.LOW
    except ValueError:
        return Severity.LOW


def _is_error(event: DecisionEvent) -> bool:
    """A recorded error counts as business loss unless the decision DENIED
    the action (a denied action never executes, so K cannot realize on it).
    ABSTAIN/VERIFY retain error semantics per their recorded outcome."""
    if event.decision == DecisionKind.DENY:
        return False
    outcome = event.observed_outcome or {}
    return outcome.get("error") is True


def _escalated(event: DecisionEvent, threshold: float) -> bool:
    p = event.calibrated_probability
    return p is not None and p >= threshold


def _loss(events: Sequence[DecisionEvent], threshold: float, profile: CostProfile) -> float:
    """Business loss of unreviewed errors + review cost of escalated cases."""
    unreviewed_errors = [
        _is_error(e) for e in events if not _escalated(e, threshold)
    ]
    unreviewed_sevs = [
        _severity(e) for e in events if not _escalated(e, threshold)
    ]
    business = severity_weighted_loss(unreviewed_errors, unreviewed_sevs, profile)
    reviews = sum(1 for e in events if _escalated(e, threshold))
    return business + reviews * profile.escalate_cost


def _review_load(events: Sequence[DecisionEvent], threshold: float) -> float:
    if not events:
        return 0.0
    return sum(1 for e in events if _escalated(e, threshold)) / len(events)


class ReplayLab:
    """Evaluates alternative escalation policies against recorded history."""

    def __init__(self, cost_profile: CostProfile):
        self.cost_profile = cost_profile

    def simulate(
        self,
        events: Sequence[DecisionEvent],
        policy: PolicyBundle,
        new_threshold: float,
    ) -> ReplayOutcome:
        """Re-decide every event at `new_threshold` vs the policy's own
        threshold. Deterministic, no model calls."""
        before_tau = policy.escalation_threshold
        before_loss = _loss(events, before_tau, self.cost_profile)
        after_loss = _loss(events, new_threshold, self.cost_profile)
        total_before = before_loss
        total_after = after_loss

        per_case: list[dict] = []
        for event in events:
            was_escalated = _escalated(event, before_tau)
            is_escalated = _escalated(event, new_threshold)
            if was_escalated != is_escalated:
                per_case.append(
                    {
                        "event_id": event.event_id,
                        "before": (
                            DecisionKind.ESCALATE.value if was_escalated else event.decision.value
                        ),
                        "after": (
                            DecisionKind.ESCALATE.value if is_escalated else event.decision.value
                        ),
                        "expected_loss": event.expected_loss,
                    }
                )

        return ReplayOutcome(
            before_loss=round(before_loss, 4),
            after_loss=round(after_loss, 4),
            before_review_load=round(_review_load(events, before_tau), 4),
            after_review_load=round(_review_load(events, new_threshold), 4),
            total_before=round(total_before, 4),
            total_after=round(total_after, 4),
            per_case_diff=per_case,
        )

    def simulate_with_ledger(
        self,
        ledger: AuditLedger,
        policy: PolicyBundle,
        new_threshold: float,
        limit: int = 500,
    ) -> ReplayOutcome:
        """Pull up to `limit` events (append order) from the ledger and simulate."""
        events = ledger.read_all(limit=limit)
        return self.simulate(events, policy, new_threshold)
