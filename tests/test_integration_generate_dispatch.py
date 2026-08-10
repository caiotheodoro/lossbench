"""Integration: generate_suite dispatches all three finance domains."""

from __future__ import annotations

import pytest

from lossbench.generate import DOMAINS, generate_suite, verifier_for


@pytest.mark.parametrize("domain", DOMAINS)
def test_generate_suite_all_domains(domain: str):
    tasks = generate_suite(domain, seed=7, n_tasks=80)
    assert len(tasks) == 80
    assert all(t.domain == domain for t in tasks)


@pytest.mark.parametrize("domain", DOMAINS)
def test_verifier_agreement_across_domains(domain: str):
    tasks = generate_suite(domain, seed=42, n_tasks=80)
    verifier = verifier_for(domain)
    for task in tasks:
        assert verifier(task, task.gold) is True


@pytest.mark.parametrize("domain", DOMAINS)
def test_determinism_across_domains(domain: str):
    a = generate_suite(domain, seed=11, n_tasks=60)
    b = generate_suite(domain, seed=11, n_tasks=60)
    assert [t.model_dump_json() for t in a] == [t.model_dump_json() for t in b]


@pytest.mark.parametrize("domain", DOMAINS)
def test_signatures_unique_across_domains(domain: str):
    tasks = generate_suite(domain, seed=3, n_tasks=100)
    sigs = {t.signature for t in tasks}
    assert len(sigs) == len(tasks)


def test_verifier_registry_complete():
    assert set(verifier_for(d) for d in DOMAINS) is not None
    for domain in DOMAINS:
        assert verifier_for(domain) is not None


def test_unknown_domain_verifier_raises():
    with pytest.raises(ValueError, match="no verifier registered"):
        verifier_for("retail")
