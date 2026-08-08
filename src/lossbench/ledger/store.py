"""Append-only audit ledger backed by DuckDB with a SHA-256 hash chain.

Concurrency contract: single-writer. A per-ledger lock serializes appends,
and the read-compute-insert sequence runs inside one transaction with a
UNIQUE constraint on seq, so interleaved writers fail loudly instead of
corrupting the chain.

Tamper-evidence contract: verify() checks BOTH prev-hash linkage and
recomputed content hashes. The chain has no external anchor (a tail
truncation or a consistent rewrite from some row onward is undetectable by
the chain alone); export_jsonl emits chain fields so an external verifier
can re-derive them, and production deployments should add a periodic signed
head checkpoint. This limitation is documented, not hidden.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import duckdb

from lossbench.schema import DecisionEvent

GENESIS = "GENESIS"

_SCHEMA_EVENTS = (
    "CREATE TABLE IF NOT EXISTS events("
    "event_id VARCHAR PRIMARY KEY, "
    "seq INTEGER UNIQUE, "
    "tenant_id VARCHAR, "
    "trajectory_id VARCHAR, "
    "task_id VARCHAR, "
    "event_json VARCHAR, "
    "prev_hash VARCHAR, "
    "chain_hash VARCHAR)"
)

_SCHEMA_META = "CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)"


def _chain_hash(event_json: str, prev_hash: str) -> str:
    """SHA-256 over the canonical event JSON joined to the previous hash."""
    return hashlib.sha256(f"{event_json}|{prev_hash}".encode()).hexdigest()


class AuditLedger:
    """Append-only store of DecisionEvents with a tamper-evident hash chain."""

    def __init__(self, path: str = ":memory:"):
        """Open a ledger; creates the backing tables when absent."""
        self._conn = duckdb.connect(path)
        self._lock = threading.Lock()
        self._conn.execute(_SCHEMA_EVENTS)
        self._conn.execute(_SCHEMA_META)
        self._conn.execute(
            "INSERT INTO meta (key, value) SELECT 'genesis', ? "
            "WHERE NOT EXISTS (SELECT 1 FROM meta WHERE key = 'genesis')",
            [GENESIS],
        )

    def append(self, event: DecisionEvent) -> str:
        """Append an event atomically and return the new chain head hash.

        Runs the existence check, seq derivation, and insert inside one
        transaction under a per-ledger lock; the UNIQUE constraint on seq
        makes any lost update fail loudly. Raises ValueError when event_id
        is already present; an event_id is immutable once written.
        """
        event_json = json.dumps(
            event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                exists = self._conn.execute(
                    "SELECT 1 FROM events WHERE event_id = ?", [event.event_id]
                ).fetchone()
                if exists is not None:
                    raise ValueError(f"event_id already exists: {event.event_id}")
                max_seq = self._conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM events"
                ).fetchone()[0]
                seq = max_seq + 1
                prev_hash = (
                    GENESIS
                    if seq == 1
                    else self._conn.execute(
                        "SELECT chain_hash FROM events WHERE seq = ?", [seq - 1]
                    ).fetchone()[0]
                )
                chain_hash = _chain_hash(event_json, prev_hash)
                self._conn.execute(
                    "INSERT INTO events (event_id, seq, tenant_id, trajectory_id, "
                    "task_id, event_json, prev_hash, chain_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        event.event_id,
                        seq,
                        event.tenant_id,
                        event.trajectory_id,
                        event.task_id,
                        event_json,
                        prev_hash,
                        chain_hash,
                    ],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return chain_hash

    def get(self, event_id: str) -> DecisionEvent | None:
        """Return the stored event, or None when event_id is absent."""
        row = self._conn.execute(
            "SELECT event_json FROM events WHERE event_id = ?", [event_id]
        ).fetchone()
        if row is None:
            return None
        return DecisionEvent.model_validate_json(row[0])

    def events_by_trajectory(self, trajectory_id: str) -> list[DecisionEvent]:
        """Return events for a trajectory in append order."""
        rows = self._conn.execute(
            "SELECT event_json FROM events WHERE trajectory_id = ? ORDER BY seq",
            [trajectory_id],
        ).fetchall()
        return [DecisionEvent.model_validate_json(row[0]) for row in rows]

    def events_by_tenant(self, tenant_id: str, limit: int = 100) -> list[DecisionEvent]:
        """Return the most recent events for a tenant, newest first."""
        rows = self._conn.execute(
            "SELECT event_json FROM events WHERE tenant_id = ? "
            "ORDER BY seq DESC LIMIT ?",
            [tenant_id, limit],
        ).fetchall()
        return [DecisionEvent.model_validate_json(row[0]) for row in rows]

    def read_all(self, limit: int = 1000) -> list[DecisionEvent]:
        """Return all events in append order (across tenants), oldest first."""
        rows = self._conn.execute(
            "SELECT event_json FROM events ORDER BY seq ASC LIMIT ?", [limit]
        ).fetchall()
        return [DecisionEvent.model_validate_json(row[0]) for row in rows]

    def verify(self) -> dict[str, Any]:
        """Recompute the chain over all rows; report validity and first bad seq."""
        rows = self._conn.execute(
            "SELECT seq, event_json, prev_hash, chain_hash FROM events ORDER BY seq"
        ).fetchall()
        prev = GENESIS
        head = GENESIS
        first_bad_seq = None
        for seq, event_json, stored_prev, stored_hash in rows:
            if stored_prev != prev or _chain_hash(event_json, prev) != stored_hash:
                first_bad_seq = seq
                break
            head = stored_hash
            prev = stored_hash
        return {
            "valid": first_bad_seq is None,
            "head": head,
            "n_events": len(rows),
            "first_bad_seq": first_bad_seq,
        }

    def export_jsonl(self, path: str | Path, tenant_id: str | None = None) -> int:
        """Write canonical JSON lines with chain fields; return count written.

        Each line is {"event": {...}, "seq": n, "prev_hash": ..., "chain_hash":
        ...} so an external verifier can re-derive the chain from the export
        without access to the ledger file.
        """
        sql = (
            "SELECT event_json, seq, prev_hash, chain_hash FROM events"
        )
        params: list[Any] = []
        if tenant_id is not None:
            sql += " WHERE tenant_id = ?"
            params.append(tenant_id)
        sql += " ORDER BY seq"
        rows = self._conn.execute(sql, params).fetchall()
        with open(path, "w", encoding="utf-8") as fh:
            for event_json, seq, prev_hash, chain_hash in rows:
                fh.write(
                    json.dumps(
                        {
                            "event": json.loads(event_json),
                            "seq": seq,
                            "prev_hash": prev_hash,
                            "chain_hash": chain_hash,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return len(rows)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
