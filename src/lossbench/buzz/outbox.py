"""Buzz collaboration projection: durable outbox of signed review events.

LossBench ledger -> outbox -> Buzz signed review event, and Buzz resolution ->
verified callback. The core has no network: publishing to a real Buzz relay is
a later adapter, and mark_published simulates delivery. resolve_callback is the
"verified callback -> ledger" seam; it validates shape and state but does NOT
append to a ledger (ReviewService owns ledger writes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb

from lossbench.hitl.review import ReviewRequest, ReviewResolution
from lossbench.schema import DecisionEvent
from lossbench.util.canonical import canonical_json

_SCHEMA_OUTBOX = (
    "CREATE TABLE IF NOT EXISTS outbox("
    "outbox_id VARCHAR PRIMARY KEY, "
    "decision_id VARCHAR, "
    "tenant_id VARCHAR, "
    "community VARCHAR, "
    "kind VARCHAR, "
    "payload VARCHAR, "
    "status VARCHAR, "
    "created_at TIMESTAMP)"
)

_PENDING = "pending"
_PUBLISHED = "published"

_KINDS = ("REVIEW_REQUEST", "REVIEW_RESOLVED")
_RESOLUTIONS = ("APPROVE", "REJECT", "AMEND")


@dataclass(frozen=True)
class BuzzEvent:
    """One outbox row projected as a signed-event view."""

    outbox_id: str
    decision_id: str
    tenant_id: str
    community: str
    kind: str
    payload: dict[str, Any]


def _outbox_id(decision_id: str, kind: str) -> str:
    """Deterministic primary key for a (decision_id, kind) pair."""
    return hashlib.sha256(f"{decision_id}|{kind}".encode()).hexdigest()[:16]


def _naive_utc(value: datetime) -> datetime:
    """Strip tzinfo for DuckDB TIMESTAMP storage; aware input is UTC."""
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def build_payload(
    event: DecisionEvent | None,
    *,
    kind: str,
    decision_id: str,
    tenant_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Signed-event payload: {"kind", "decision_id", "tenant_id",
    "payload_hash", "extra"}. payload_hash is the SHA-256 hex digest of the
    canonical JSON of extra; extra defaults to {}. Deterministic: equal inputs
    produce equal payloads regardless of dict insertion order. event is
    reserved for deriving ledger-sourced review fields in future versions and
    is unused today; the outbox stamps the tenant's "community" on the
    returned payload before storage (one tenant -> one community)."""
    return {
        "kind": kind,
        "decision_id": decision_id,
        "tenant_id": tenant_id,
        "payload_hash": hashlib.sha256(
            canonical_json(extra or {}).encode()
        ).hexdigest(),
        "extra": extra or {},
    }


class BuzzOutbox:
    """Durable outbox of signed review events, backed by DuckDB."""

    def __init__(self, path: str = ":memory:"):
        """Open an outbox; creates the backing table when absent."""
        self._conn = duckdb.connect(path)
        self._conn.execute(_SCHEMA_OUTBOX)

    def enqueue_review_request(self, request: ReviewRequest, community: str) -> BuzzEvent:
        """Append an outbox row (status 'pending') for kind REVIEW_REQUEST.

        Builds the signed-event payload via build_payload and returns the
        BuzzEvent. Idempotent per decision_id: a second enqueue of the same
        decision_id returns the existing row instead of duplicating.
        """
        kind = "REVIEW_REQUEST"
        existing = self.event_for(request.decision_id, kind)
        if existing is not None:
            return existing
        payload = build_payload(
            None,
            kind=kind,
            decision_id=request.decision_id,
            tenant_id=request.tenant_id,
            extra={
                "trajectory_id": request.trajectory_id,
                "task_id": request.task_id,
                "proposed_action": request.proposed_action,
                "expected_loss": request.expected_loss,
                "rationale": request.rationale,
                "policy_ref": request.policy_ref,
                "sla_seconds": request.sla_seconds,
                "required_role": request.required_role,
            },
        )
        payload["community"] = community
        self._insert(
            request.decision_id, request.tenant_id, community, kind, payload, request.created_at
        )
        return self.event_for(request.decision_id, kind)

    def enqueue_resolution(self, resolution: ReviewResolution, community: str) -> BuzzEvent:
        """Append an outbox row (status 'pending') for kind REVIEW_RESOLVED.

        The tenant is read from the decision's pending-or-published
        REVIEW_REQUEST row, which must exist; raises ValueError otherwise.
        Idempotent per decision_id: a second enqueue of the same decision_id
        returns the existing row instead of duplicating.
        """
        kind = "REVIEW_RESOLVED"
        existing = self.event_for(resolution.decision_id, kind)
        if existing is not None:
            return existing
        request_row = self._conn.execute(
            "SELECT tenant_id FROM outbox WHERE decision_id = ? AND kind = 'REVIEW_REQUEST'",
            [resolution.decision_id],
        ).fetchone()
        if request_row is None:
            raise ValueError(f"no outbox review request for {resolution.decision_id}")
        payload = build_payload(
            None,
            kind=kind,
            decision_id=resolution.decision_id,
            tenant_id=request_row[0],
            extra={
                "resolution": resolution.resolution,
                "reviewer": resolution.reviewer,
                "amended_action": resolution.amended_action,
                "note": resolution.note,
            },
        )
        payload["community"] = community
        self._insert(
            resolution.decision_id, request_row[0], community, kind, payload, resolution.resolved_at
        )
        return self.event_for(resolution.decision_id, kind)

    def mark_published(self, outbox_id: str) -> None:
        """Set the row's status to 'published'; raises ValueError when the
        outbox_id is unknown."""
        if self._conn.execute(
            "SELECT 1 FROM outbox WHERE outbox_id = ?", [outbox_id]
        ).fetchone() is None:
            raise ValueError(f"unknown outbox_id: {outbox_id}")
        self._conn.execute(
            "UPDATE outbox SET status = 'published' WHERE outbox_id = ?", [outbox_id]
        )

    def pending(self) -> list[BuzzEvent]:
        """Return all rows with status 'pending', ordered by created_at then
        outbox_id."""
        return self._select("WHERE status = ?", [_PENDING])

    def published(self) -> list[BuzzEvent]:
        """Return all rows with status 'published', ordered by created_at then
        outbox_id."""
        return self._select("WHERE status = ?", [_PUBLISHED])

    def event_for(self, decision_id: str, kind: str) -> BuzzEvent | None:
        """Return the row for a (decision_id, kind) pair, or None."""
        rows = self._select(
            "WHERE decision_id = ? AND kind = ?", [decision_id, kind], limit=True
        )
        return rows[0] if rows else None

    def resolve_callback(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle a verified resolution callback from Buzz.

        The payload must contain decision_id, a resolution in
        (APPROVE, REJECT, AMEND) and a reviewer. The corresponding
        REVIEW_REQUEST outbox row must exist and be published, and the
        REVIEW_RESOLVED row must exist; raises ValueError with a clear
        message otherwise. Marks the resolution row published and returns
        {"accepted": True, "outbox_id": <resolution outbox_id>}. This seam
        validates state only; ledger writes stay with ReviewService.
        Replays of an already-published resolution are accepted idempotently.
        """
        decision_id = payload.get("decision_id")
        resolution = payload.get("resolution")
        reviewer = payload.get("reviewer")
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("callback payload missing decision_id")
        if resolution not in _RESOLUTIONS:
            raise ValueError(f"invalid resolution: {resolution!r}; expected one of {_RESOLUTIONS}")
        if not isinstance(reviewer, str) or not reviewer:
            raise ValueError("callback payload missing reviewer")
        request = self.event_for(decision_id, "REVIEW_REQUEST")
        if request is None:
            raise ValueError(f"no outbox review request for {decision_id}")
        status = self._conn.execute(
            "SELECT status FROM outbox WHERE outbox_id = ?", [request.outbox_id]
        ).fetchone()[0]
        if status != _PUBLISHED:
            raise ValueError(f"review request for {decision_id} is not published")
        resolved = self.event_for(decision_id, "REVIEW_RESOLVED")
        if resolved is None:
            raise ValueError(f"no outbox resolution for {decision_id}")
        self.mark_published(resolved.outbox_id)
        return {"accepted": True, "outbox_id": resolved.outbox_id}

    def _insert(
        self,
        decision_id: str,
        tenant_id: str,
        community: str,
        kind: str,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        """Insert one outbox row with status 'pending'."""
        self._conn.execute(
            "INSERT INTO outbox "
            "(outbox_id, decision_id, tenant_id, community, kind, payload, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                _outbox_id(decision_id, kind),
                decision_id,
                tenant_id,
                community,
                kind,
                json.dumps(payload),
                _PENDING,
                _naive_utc(created_at),
            ],
        )

    def _select(
        self, where: str, params: list[Any], limit: bool = False
    ) -> list[BuzzEvent]:
        """Read rows into BuzzEvent views, ordered by created_at then outbox_id."""
        sql = (
            "SELECT outbox_id, decision_id, tenant_id, community, kind, payload "
            f"FROM outbox {where} ORDER BY created_at, outbox_id"
            + (" LIMIT 1" if limit else "")
        )
        return [
            BuzzEvent(
                outbox_id=row[0],
                decision_id=row[1],
                tenant_id=row[2],
                community=row[3],
                kind=row[4],
                payload=json.loads(row[5]),
            )
            for row in self._conn.execute(sql, params).fetchall()
        ]
