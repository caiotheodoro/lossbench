"""P1.1 reconciliation generator acceptance tests."""

from __future__ import annotations

import pytest

from lossbench.generate import DOMAINS, generate_suite
from lossbench.generate.reconciliation import (
    EXCEPTION_TYPES,
    SEVERITY_BY_EXCEPTION,
    compute_verdict_from_state,
    verifier_reconciliation,
)
from lossbench.generate.taxonomy import task_signature


def test_same_seed_byte_identical():
    a = generate_suite("reconciliation", seed=7, n_tasks=50)
    b = generate_suite("reconciliation", seed=7, n_tasks=50)
    assert [t.model_dump_json() for t in a] == [t.model_dump_json() for t in b]


def test_verifier_agrees_with_gold_100pct():
    for seed in (7, 42, 777):
        tasks = generate_suite("reconciliation", seed=seed, n_tasks=100)
        for task in tasks:
            outcome = {
                "verdict": task.gold["verdict"],
                "exception_type": task.gold["exception_type"],
            }
            assert verifier_reconciliation(task, outcome) is True


def test_verifier_rejects_wrong_outcome():
    tasks = generate_suite("reconciliation", seed=7, n_tasks=300)
    exceptions = [t for t in tasks if t.gold["verdict"] == "EXCEPTION"]
    assert exceptions, "expected exceptions in the suite"
    for task in exceptions[:40]:
        wrong = {"verdict": "MATCH", "exception_type": None}
        assert verifier_reconciliation(task, wrong) is False


def test_verifier_does_not_read_gold():
    tasks = generate_suite("reconciliation", seed=42, n_tasks=100)
    for task in tasks[:60]:
        expected = compute_verdict_from_state(task.initial_state)
        assert expected == task.gold, "verifier independent of gold label"


def test_severity_mix_honored():
    tasks = generate_suite(
        "reconciliation",
        seed=11,
        n_tasks=400,
        severity_mix={"HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1},
    )
    exceptions = [t for t in tasks if t.gold["verdict"] == "EXCEPTION"]
    assert len(exceptions) > 100
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for task in exceptions:
        counts[task.severity.value] += 1
    observed = {k: v / len(exceptions) for k, v in counts.items()}
    assert abs(observed["HIGH"] - 0.6) <= 0.05
    assert abs(observed["MEDIUM"] - 0.3) <= 0.05
    assert abs(observed["LOW"] - 0.1) <= 0.05


def test_all_nine_classes_appear():
    tasks = generate_suite("reconciliation", seed=7, n_tasks=500)
    present = {t.gold["exception_type"] for t in tasks if t.gold["exception_type"]}
    assert set(EXCEPTION_TYPES) <= present


def test_signatures_distinct_and_stable():
    tasks = generate_suite("reconciliation", seed=3, n_tasks=120)
    sigs = {t.signature for t in tasks}
    assert len(sigs) == len(tasks)
    assert sigs == {task_signature(t) for t in tasks}


def test_all_tasks_pass_domain_verifier():
    for seed in (1, 2, 3):
        tasks = generate_suite("reconciliation", seed=seed, n_tasks=200)
        for task in tasks:
            outcome = {
                "verdict": task.gold["verdict"],
                "exception_type": task.gold["exception_type"],
            }
            assert verifier_reconciliation(task, outcome) is True


def test_severity_taxonomy_complete():
    assert set(SEVERITY_BY_EXCEPTION) == set(EXCEPTION_TYPES)


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown domain"):
        generate_suite("retail", seed=1, n_tasks=5)


def test_domains_constant():
    assert DOMAINS == ("reconciliation", "payment_repair", "settlement")
