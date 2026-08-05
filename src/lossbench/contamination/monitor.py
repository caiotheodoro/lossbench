"""Signature-based contamination detection between train and eval task sets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from lossbench.schema import Task

_METADATA_FIELDS = frozenset({"id", "seed", "signature"})


def task_signature(task: Task) -> str:
    """SHA-256 hex over sorted (field, value) pairs of content fields, metadata excluded."""
    payload = json.dumps(
        task.model_dump(exclude=_METADATA_FIELDS),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _signatures(tasks: Sequence[Task]) -> set[str]:
    return {task_signature(t) for t in tasks}


def signature_overlap(train: Sequence[Task], eval_set: Sequence[Task]) -> float:
    """|sig(train) ∩ sig(eval)| / |sig(train)|; 0.0 when train is empty."""
    if not train:
        return 0.0
    train_sigs = _signatures(train)
    eval_sigs = _signatures(eval_set)
    return len(train_sigs & eval_sigs) / len(train_sigs)


def leak_fraction_detected(
    train: Sequence[Task],
    eval_set: Sequence[Task],
    leaked: Sequence[Task],
    threshold: float = 0.0,
) -> float:
    """Fraction of leaked tasks detected: detected iff signature is in train or in eval_set."""
    if not leaked:
        return 0.0
    train_sigs = _signatures(train)
    eval_sigs = _signatures(eval_set)
    detected = 0
    for task in leaked:
        sig = task_signature(task)
        if sig in train_sigs or sig in eval_sigs:
            detected += 1
    return detected / len(leaked)


def monitor_report(
    train: Sequence[Task],
    eval_set: Sequence[Task],
    leaked: Sequence[Task] | None = None,
) -> dict[str, float]:
    """Return overlap, false_fire, and detection for the train/eval pair."""
    overlap = signature_overlap(train, eval_set)
    false_fire = 1.0 if overlap == 0.0 else 0.0
    detection = (
        leak_fraction_detected(train, eval_set, leaked) if leaked is not None else 1.0
    )
    return {"overlap": overlap, "false_fire": false_fire, "detection": detection}
