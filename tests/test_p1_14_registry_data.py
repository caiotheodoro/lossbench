import pytest
import yaml

from lossbench.costs import registry_data as rd
from lossbench.costs.registry_data import (
    known_event_types,
    load_registry,
    profile_from_registry,
)
from lossbench.schema import CostProfile, CostSource

MIN_ENTRIES = 8
MIN_DOMAINS = 4
ALL_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_registry_has_at_least_8_entries():
    registry = load_registry()
    assert len(registry) >= MIN_ENTRIES
    assert len({entry.domain for entry in registry.values()}) >= MIN_DOMAINS


def test_all_entries_have_ranges():
    for entry in load_registry().values():
        assert entry.cost_low <= entry.cost_typical <= entry.cost_high


def test_all_entries_have_sources():
    raw = yaml.safe_load(rd.REGISTRY_PATH.read_text())
    source_map = raw["sources"]
    for entry in load_registry().values():
        assert len(entry.sources) >= 1
        for source_id in entry.sources:
            assert source_id in source_map
            assert source_map[source_id]["url"].startswith("http")


def test_duplicate_id_rejected():
    registry = load_registry()
    assert len(set(registry)) == len(registry)


def test_missing_required_field_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "registry.yaml"
    bad.write_text(
        "entries:\n"
        "  broken:\n"
        "    event_type: broken\n"
        "    cost_low: 1.0\n"
    )
    monkeypatch.setattr(rd, "REGISTRY_PATH", bad)
    with pytest.raises(ValueError, match="missing required fields"):
        rd.load_registry()


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError, match="unknown event_type"):
        profile_from_registry("does_not_exist")


def test_profile_from_registry_builds_costprofile():
    entry = load_registry()["misposted_ach"]
    profile = profile_from_registry(entry.event_type)
    assert isinstance(profile, CostProfile)
    assert set(profile.severity_costs) == ALL_SEVERITIES
    assert profile.severity_costs["LOW"] == pytest.approx(entry.cost_typical * 0.1)
    assert profile.severity_costs["MEDIUM"] == pytest.approx(entry.cost_typical * 0.5)
    assert profile.severity_costs["HIGH"] == pytest.approx(entry.cost_typical)
    assert profile.severity_costs["CRITICAL"] == pytest.approx(entry.cost_typical * 10.0)
    assert all(isinstance(source, CostSource) for source in profile.sources)
    assert profile.id == "reconciliation"


def test_profile_from_registry_scaling_and_profile_id():
    entry = load_registry()["misrouted_wire"]
    profile = profile_from_registry(entry.event_type, profile_id="wire_test", scale=2.0)
    assert profile.id == "wire_test"
    assert profile.severity_costs["HIGH"] == pytest.approx(entry.cost_typical * 2.0)
    assert profile.severity_costs["LOW"] == pytest.approx(entry.cost_typical * 2.0 * 0.1)
    assert profile.severity_costs["CRITICAL"] == pytest.approx(
        entry.cost_typical * 2.0 * 10.0
    )


def test_known_event_types_nonempty():
    event_types = known_event_types()
    assert event_types
    assert event_types == sorted(set(event_types))
    assert all(isinstance(event_type, str) for event_type in event_types)


def test_deterministic():
    assert load_registry() == load_registry()
