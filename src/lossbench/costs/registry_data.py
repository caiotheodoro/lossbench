"""Empirical severity-cost registry.

Loads order-of-magnitude public cost anchors from data/registry.yaml and
builds CostProfiles from them. The figures are sourced, contested anchors,
not actuarial values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from lossbench.schema import CostProfile, CostSource

REGISTRY_PATH = Path(__file__).parent / "data" / "registry.yaml"

_LOW_FACTOR = 0.1
_MEDIUM_FACTOR = 0.5
_HIGH_FACTOR = 1.0
_CRITICAL_FACTOR = 10.0


@dataclass(frozen=True)
class EmpiricalCost:
    """One sourced, order-of-magnitude cost anchor for an event type."""

    event_type: str
    domain: str
    cost_low: float
    cost_typical: float
    cost_high: float
    unit: str
    sources: tuple[str, ...]


def load_registry() -> dict[str, EmpiricalCost]:
    """Parse data/registry.yaml into id -> EmpiricalCost.

    Raises ValueError on duplicate ids, missing required fields, or
    non-monotonic cost ranges.
    """
    raw = yaml.safe_load(REGISTRY_PATH.read_text())
    result: dict[str, EmpiricalCost] = {}
    for entry_id, fields in raw.get("entries", {}).items():
        if entry_id in result:
            raise ValueError(f"duplicate registry entry id '{entry_id}'")
        try:
            entry = EmpiricalCost(
                event_type=fields["event_type"],
                domain=fields["domain"],
                cost_low=float(fields["cost_low"]),
                cost_typical=float(fields["cost_typical"]),
                cost_high=float(fields["cost_high"]),
                unit=fields["unit"],
                sources=tuple(fields["sources"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"registry entry '{entry_id}' missing required fields"
            ) from exc
        if not (entry.cost_low <= entry.cost_typical <= entry.cost_high):
            raise ValueError(
                f"registry entry '{entry_id}' has non-monotonic cost range"
            )
        result[entry_id] = entry
    return result


def profile_from_registry(
    event_type: str, profile_id: str = "reconciliation", scale: float = 1.0
) -> CostProfile:
    """Build a CostProfile from the registry entry for `event_type`.

    Severity costs are LOW = typical * 0.1, MEDIUM = typical * 0.5,
    HIGH = typical, CRITICAL = typical * 10, all scaled by `scale`.
    Raises ValueError for unknown event types or source ids.
    """
    raw = yaml.safe_load(REGISTRY_PATH.read_text())
    source_map = {
        source_id: CostSource(**fields)
        for source_id, fields in raw.get("sources", {}).items()
    }
    fields = _find_entry(raw, event_type)
    typical = float(fields["cost_typical"])
    try:
        sources = [source_map[source_id] for source_id in fields["sources"]]
    except KeyError as exc:
        raise ValueError(f"unknown source id '{exc.args[0]}'") from exc
    return CostProfile(
        id=profile_id,
        description=(
            f"Severity-cost profile derived from registry entry '{event_type}' "
            f"({fields['domain']}, {fields['unit']}), scaled by {scale}."
        ),
        sources=sources,
        severity_costs={
            "LOW": typical * _LOW_FACTOR * scale,
            "MEDIUM": typical * _MEDIUM_FACTOR * scale,
            "HIGH": typical * _HIGH_FACTOR * scale,
            "CRITICAL": typical * _CRITICAL_FACTOR * scale,
        },
    )


def known_event_types() -> list[str]:
    """Sorted unique event types present in the shipped registry."""
    return sorted({entry.event_type for entry in load_registry().values()})


def _find_entry(raw: dict, event_type: str) -> dict:
    for fields in raw.get("entries", {}).values():
        if fields.get("event_type") == event_type:
            return fields
    raise ValueError(f"unknown event_type '{event_type}' in registry")
