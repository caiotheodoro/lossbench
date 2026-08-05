"""Canonical task signature: SHA-256 over content fields, metadata excluded.

The signature definition is shared by the contamination monitor (P1.2); the
two implementations must agree on what counts as content vs metadata.
"""

from __future__ import annotations

import hashlib
import json

from lossbench.schema import Task

_METADATA_FIELDS = {"id", "seed", "signature"}


def task_signature(task: Task) -> str:
    """SHA-256 hex over the task's content fields, metadata excluded.

    Content = everything except metadata (id, seed, signature): the prompt,
    initial_state, gold, severity, domain, difficulty, tools, verifier,
    policy, cost model. Order-insensitive by construction. The serialization
    MUST match the contamination monitor's computation over the same model
    dump (a single canonical space), or stored signatures and monitor
    signatures silently diverge.
    """
    data = task.model_dump(mode="json")
    for key in _METADATA_FIELDS:
        data.pop(key, None)
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
