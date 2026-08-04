"""Deterministic canonicalization of JSON-serializable objects."""

import json
from typing import Any


class _FloatRepr(float):
    __slots__ = ()

    def __repr__(self) -> str:
        return float.__repr__(self)


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        ordered = sorted(obj.items(), key=lambda item: str(item[0]))
        return {str(k): _canonical(v) for k, v in ordered}
    if isinstance(obj, (list, tuple)):
        return [_canonical(item) for item in obj]
    if isinstance(obj, float):
        return _FloatRepr(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys recursively, compact separators,
    floats normalized via repr(). Must be stable across dict insertion
    orders and equal for equal semantic content."""
    return json.dumps(
        _canonical(obj),
        separators=(",", ":"),
        sort_keys=True,
    )


def freeze_list(obj: list[Any]) -> list[Any]:
    """Deep-freeze a list of JSON-serializable objects for membership tests:
    returns a list of canonical JSON strings."""
    frozen: list[str] = []
    seen: set[str] = set()
    for item in obj:
        key = canonical_json(item)
        if key not in seen:
            seen.add(key)
            frozen.append(key)
    return frozen
