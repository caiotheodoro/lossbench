"""Determinism utilities: hashing, RNG seeding, and CI determinism gates."""

import hashlib
import random
from typing import Any

from lossbench.schema import DecisionEvent
from lossbench.util.canonical import canonical_json


def sha256_hex(content: bytes | str) -> str:
    """SHA-256 hex digest of bytes or UTF-8 encoded text."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def hash_event(event: DecisionEvent, *, exclude: frozenset[str] = frozenset()) -> str:
    """sha256 of canonical_json(event.model_dump(mode="json")) minus excluded
    top-level keys. Used for evidence/prompt/input hashes."""
    data: dict[str, Any] = event.model_dump(mode="json")
    payload = {key: value for key, value in data.items() if key not in exclude}
    return sha256_hex(canonical_json(payload))


def seed_rng(seed: int) -> random.Random:
    """Wrap random.Random(seed); reserved for a future global seed policy."""
    return random.Random(seed)


def assert_identical(a: str, b: str, context: str = "") -> None:
    """Raise AssertionError with context if a != b (for CI determinism gates)."""
    if a == b:
        return
    prefix = f"{context}: " if context else ""
    raise AssertionError(f"{prefix}{a!r} != {b!r}")
