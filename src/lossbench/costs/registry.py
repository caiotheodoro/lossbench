"""Cost profile registry: load and list versioned CostProfiles from YAML.

Profiles are swappable inputs, never hidden constants. Ships with the four
canonical profiles in `profiles/`; users may supply their own via `load_cost_profile`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lossbench.schema import CostProfile, CostSource

PROFILES_DIR = Path(__file__).parent / "profiles"
ProfileId = str


def list_cost_profiles() -> list[str]:
    """IDs of the shipped cost profiles."""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_cost_profile(profile_id: ProfileId) -> CostProfile:
    """Load a shipped profile by id (e.g. 'flat', 'reconciliation')."""
    path = PROFILES_DIR / f"{profile_id}.yaml"
    if not path.exists():
        raise ValueError(
            f"unknown cost profile '{profile_id}'; available: {list_cost_profiles()}"
        )
    return _from_yaml(path)


def _from_yaml(path: Path) -> CostProfile:
    raw = yaml.safe_load(path.read_text())
    sources = [CostSource(**s) for s in raw.get("sources", [])]
    raw.pop("sources", None)
    return CostProfile(**raw, sources=sources)
