import pytest

from lossbench.costs.registry import list_cost_profiles, load_cost_profile
from lossbench.schema import Severity

EXPECTED = {"flat", "reconciliation", "principal_risk", "review_heavy"}


def test_list_cost_profiles_contains_canonical_four():
    assert EXPECTED <= set(list_cost_profiles())


def test_load_flat_profile():
    flat = load_cost_profile("flat")
    assert flat.id == "flat"
    assert {s.value for s in Severity} == set(flat.severity_costs)
    assert len(set(flat.severity_costs.values())) == 1


def test_load_reconciliation_profile_ordering():
    rec = load_cost_profile("reconciliation")
    assert (
        rec.severity_costs["LOW"]
        < rec.severity_costs["MEDIUM"]
        < rec.severity_costs["HIGH"]
        < rec.severity_costs["CRITICAL"]
    )
    assert rec.severity_costs["HIGH"] == 10.0


def test_load_principal_risk_has_fat_tail():
    pr = load_cost_profile("principal_risk")
    assert pr.severity_costs["CRITICAL"] >= 100 * pr.severity_costs["HIGH"]


def test_load_review_heavy_expensive_review():
    rh = load_cost_profile("review_heavy")
    assert rh.escalate_cost >= 10.0


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown cost profile"):
        load_cost_profile("does-not-exist")


def test_profiles_roundtrip_through_schema():
    for pid in list_cost_profiles():
        profile = load_cost_profile(pid)
        restored = type(profile).model_validate(profile.model_dump())
        assert restored == profile


def test_all_profiles_have_all_severities():
    for pid in list_cost_profiles():
        profile = load_cost_profile(pid)
        assert set(profile.severity_costs) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
