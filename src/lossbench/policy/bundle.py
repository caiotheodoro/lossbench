"""YAML persistence for PolicyBundle: load with validation, dump as model_dump()."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from lossbench.costs.registry import load_cost_profile
from lossbench.schema import PolicyBundle


def load_policy(path: str | Path) -> PolicyBundle:
    """Load and validate a PolicyBundle from YAML at `path`.

    The YAML schema is exactly PolicyBundle.model_dump(). Unknown fields,
    malformed YAML, and missing profile references raise ValueError.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read policy yaml '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"policy yaml '{path}' must contain a mapping, got {type(raw).__name__}"
        )
    unknown = set(raw) - set(PolicyBundle.model_fields)
    if unknown:
        raise ValueError(
            f"unknown fields in policy yaml '{path}': {sorted(unknown)}"
        )
    try:
        bundle = PolicyBundle.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid policy yaml '{path}': {exc}") from exc
    try:
        load_cost_profile(bundle.cost_model_id)
    except ValueError as exc:
        raise ValueError(
            f"policy '{path}' references unknown cost_model_id "
            f"'{bundle.cost_model_id}': {exc}"
        ) from exc
    return bundle


def dump_policy(bundle: PolicyBundle, path: str | Path) -> None:
    """Write `bundle` to `path` as YAML in model_dump() field order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(bundle.model_dump(), sort_keys=False))
