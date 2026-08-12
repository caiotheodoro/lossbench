"""HITL review service: a thin durable-state layer over the audit ledger.

All review state lives in the ledger as DecisionEvents; the service is
stateless beyond its ledger reference. Review open markers are recorded on
events whose rationale starts with "[review-opened] " and resolved markers on
events whose rationale starts with "[review-resolved] ". Temporal
orchestration and timers are a later integration; SLA checks are computed on
demand from ledger data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from lossbench.ledger import AuditLedger
from lossbench.schema import DecisionEvent, DecisionKind

_OPEN_PREFIX = "[review-opened] "
_RESOLVED_PREFIX = "[review-resolved] "
_DEFAULT_SLA_SECONDS = 28800
_DEFAULT_REQUIRED_ROLE = "analyst"
_SCAN_LIMIT = 1_000_000


def _utcnow() -> datetime:
    """Return the current time as a tz-aware UTC datetime."""
    return datetime.now(UTC)


@dataclass
class ReviewRequest:
    """A human review opened against an ESCALATE decision."""

    decision_id: str
    trajectory_id: str
    tenant_id: str
    task_id: str
    proposed_action: dict[str, Any]
    expected_loss: float
    rationale: str
    policy_ref: str
    sla_seconds: int = _DEFAULT_SLA_SECONDS
    required_role: str = _DEFAULT_REQUIRED_ROLE
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class ReviewResolution:
    """A reviewer's disposition of one open review."""

    decision_id: str
    reviewer: str
    resolution: str
    amended_action: dict[str, Any] | None = None
    note: str = ""
    resolved_at: datetime = field(default_factory=_utcnow)


class ReviewService:
    """Open, resolve and inspect human reviews, durably in the ledger."""

    def __init__(self, ledger: AuditLedger, tenant_id: str = "default"):
        """Bind the service to a ledger and a default tenant."""
        self._ledger = ledger
        self._tenant_id = tenant_id

    def open_review(self, event: DecisionEvent) -> ReviewRequest:
        """Create a ReviewRequest from an ESCALATE event and record it.

        Raises ValueError when event.decision is not ESCALATE. The recorded
        REVIEW_OPENED event reuses the decision id with a "@review-opened"
        suffix, keeps decision=ESCALATE, prefixes the rationale with the open
        marker, links parent_event_id to the decision, and carries
        decision_id, sla_seconds and required_role in observed_outcome. The
        ledger append happens before the request is built, so a failed append
        raises and no request is returned.
        """
        if event.decision != DecisionKind.ESCALATE:
            raise ValueError(f"reviews only open on ESCALATE, got {event.decision}")
        now = _utcnow()
        decision_id = event.event_id
        opened = DecisionEvent(
            event_id=f"{decision_id}@review-opened",
            tenant_id=event.tenant_id,
            trace_id=event.trace_id,
            trajectory_id=event.trajectory_id,
            task_id=event.task_id,
            parent_event_id=decision_id,
            timestamp=now,
            input_snapshot_hash=event.input_snapshot_hash,
            prompt_hash=event.prompt_hash,
            model_id=event.model_id,
            model_revision=event.model_revision,
            harness_id=event.harness_id,
            harness_revision=event.harness_revision,
            reasoning_effort=event.reasoning_effort,
            tool_name=event.tool_name,
            proposed_action=event.proposed_action,
            observed_outcome={
                "decision_id": decision_id,
                "sla_seconds": _DEFAULT_SLA_SECONDS,
                "required_role": _DEFAULT_REQUIRED_ROLE,
            },
            risk_features=event.risk_features,
            calibrated_probability=event.calibrated_probability,
            expected_loss=event.expected_loss,
            decision=DecisionKind.ESCALATE,
            rationale=f"{_OPEN_PREFIX}{event.rationale}",
            policy_id=event.policy_id,
            policy_revision=event.policy_revision,
            cost_model_id=event.cost_model_id,
            token_usage=event.token_usage,
            latency_ms=event.latency_ms,
            model_cost=event.model_cost,
            judge_cost=event.judge_cost,
            human_cost=event.human_cost,
            evidence_hash=event.evidence_hash,
            created_at=now,
        )
        self._ledger.append(opened)
        return ReviewRequest(
            decision_id=decision_id,
            trajectory_id=event.trajectory_id,
            tenant_id=event.tenant_id,
            task_id=event.task_id,
            proposed_action=event.proposed_action or {},
            expected_loss=event.expected_loss if event.expected_loss is not None else 0.0,
            rationale=event.rationale,
            policy_ref=event.policy_id,
            sla_seconds=_DEFAULT_SLA_SECONDS,
            required_role=_DEFAULT_REQUIRED_ROLE,
            created_at=now,
        )

    def resolve(self, resolution: ReviewResolution) -> DecisionEvent:
        """Validate and record a resolution; return the stored event.

        Raises ValueError for an unknown resolution value, for AMEND without
        amended_action, or when the decision has no open review in the ledger.
        The recorded REVIEW_RESOLVED event reuses the decision id with a
        "@review-resolved" suffix, maps APPROVE/REJECT/AMEND to
        ALLOW/DENY/VERIFY, links parent_event_id to the decision, and carries
        the resolution, reviewer and note in observed_outcome.
        """
        if resolution.resolution not in ("APPROVE", "REJECT", "AMEND"):
            raise ValueError(f"invalid resolution: {resolution.resolution}")
        if resolution.resolution == "AMEND" and resolution.amended_action is None:
            raise ValueError("AMEND requires amended_action")
        source = self._ledger.get(f"{resolution.decision_id}@review-opened")
        if source is None:
            raise ValueError(f"no open review for {resolution.decision_id}")
        resolved = DecisionEvent(
            event_id=f"{resolution.decision_id}@review-resolved",
            tenant_id=source.tenant_id,
            trace_id=source.trace_id,
            trajectory_id=source.trajectory_id,
            task_id=source.task_id,
            parent_event_id=resolution.decision_id,
            timestamp=resolution.resolved_at,
            input_snapshot_hash=source.input_snapshot_hash,
            prompt_hash=source.prompt_hash,
            model_id=source.model_id,
            model_revision=source.model_revision,
            harness_id=source.harness_id,
            harness_revision=source.harness_revision,
            reasoning_effort=source.reasoning_effort,
            tool_name=source.tool_name,
            proposed_action=(
                resolution.amended_action
                if resolution.resolution == "AMEND"
                else source.proposed_action
            ),
            observed_outcome={
                "resolution": resolution.resolution,
                "reviewer": resolution.reviewer,
                "note": resolution.note,
            },
            risk_features=source.risk_features,
            calibrated_probability=source.calibrated_probability,
            expected_loss=source.expected_loss,
            decision={
                "APPROVE": DecisionKind.ALLOW,
                "REJECT": DecisionKind.DENY,
                "AMEND": DecisionKind.VERIFY,
            }[resolution.resolution],
            rationale=f"{_RESOLVED_PREFIX}{resolution.note}",
            policy_id=source.policy_id,
            policy_revision=source.policy_revision,
            cost_model_id=source.cost_model_id,
            token_usage=source.token_usage,
            latency_ms=source.latency_ms,
            model_cost=source.model_cost,
            judge_cost=source.judge_cost,
            human_cost=source.human_cost,
            evidence_hash=source.evidence_hash,
            created_at=resolution.resolved_at,
        )
        self._ledger.append(resolved)
        return resolved

    def pending(self, tenant_id: str | None = None) -> list[ReviewRequest]:
        """Return open reviews with no recorded REVIEW_RESOLVED child.

        Reviews are derived from the ledger by scanning the tenant's events
        for REVIEW_OPENED markers (rationale prefixed with the open marker)
        whose decision_id has no REVIEW_RESOLVED child (a resolved-marker
        event with parent_event_id == decision_id). Ordered by created_at,
        then decision_id. tenant_id defaults to the service tenant.
        """
        tenant = tenant_id or self._tenant_id
        events = self._ledger.events_by_tenant(tenant, limit=_SCAN_LIMIT)
        resolved = {
            e.parent_event_id
            for e in events
            if e.parent_event_id is not None and e.rationale.startswith(_RESOLVED_PREFIX)
        }
        requests = []
        for event in events:
            if not event.rationale.startswith(_OPEN_PREFIX):
                continue
            request = self._request_from_event(event)
            if request.decision_id in resolved:
                continue
            requests.append(request)
        return sorted(requests, key=lambda r: (r.created_at, r.decision_id))

    def sla_overdue(self, now: datetime | None = None) -> list[ReviewRequest]:
        """Return pending reviews for the service tenant past their SLA.

        A review is overdue when created_at + sla_seconds is earlier than now;
        now defaults to the current UTC time.
        """
        now = now or _utcnow()
        return [
            request
            for request in self.pending(self._tenant_id)
            if request.created_at + timedelta(seconds=request.sla_seconds) < now
        ]

    def _request_from_event(self, event: DecisionEvent) -> ReviewRequest:
        """Rebuild a ReviewRequest from its stored REVIEW_OPENED event."""
        outcome = event.observed_outcome or {}
        rationale = event.rationale
        if rationale.startswith(_OPEN_PREFIX):
            rationale = rationale[len(_OPEN_PREFIX) :]
        return ReviewRequest(
            decision_id=outcome.get("decision_id") or event.parent_event_id or event.event_id,
            trajectory_id=event.trajectory_id,
            tenant_id=event.tenant_id,
            task_id=event.task_id,
            proposed_action=event.proposed_action or {},
            expected_loss=event.expected_loss if event.expected_loss is not None else 0.0,
            rationale=rationale,
            policy_ref=event.policy_id,
            sla_seconds=outcome.get("sla_seconds", _DEFAULT_SLA_SECONDS),
            required_role=outcome.get("required_role", _DEFAULT_REQUIRED_ROLE),
            created_at=event.created_at,
        )
