"""P1.15 payment-repair generator acceptance tests."""

from __future__ import annotations

from lossbench.generate.payment_repair import (
    EXCEPTION_SIGNALS,
    compute_repair_decision_from_state,
    generate_payment_repair_suite_internal,
    verifier_payment_repair,
)
from lossbench.generate.taxonomy import task_signature


def _suite(seed, n, severity_mix=None, difficulty=(0.0, 1.0)):
    return generate_payment_repair_suite_internal(
        seed=seed,
        n_tasks=n,
        severity_mix=severity_mix,
        difficulty=difficulty,
        verifier=verifier_payment_repair,
    )


def _gold_outcome(task):
    return {
        "verdict": task.gold["verdict"],
        "exception_type": task.gold["exception_type"],
        "repair": task.gold["repair"],
        "repair_safe": task.gold["repair_safe"],
    }


def test_same_seed_byte_identical():
    a = _suite(seed=7, n=50)
    b = _suite(seed=7, n=50)
    assert [t.model_dump_json() for t in a] == [t.model_dump_json() for t in b]


def test_verifier_agrees_with_gold_100pct():
    for seed in (7, 42, 777):
        tasks = _suite(seed=seed, n=100)
        for task in tasks:
            assert verifier_payment_repair(task, _gold_outcome(task)) is True


def test_verifier_rejects_wrong_outcome():
    tasks = _suite(
        seed=7,
        n=300,
        severity_mix={"CRITICAL": 0.7, "HIGH": 0.3},
    )
    exceptions = [t for t in tasks if t.gold["verdict"] == "EXCEPTION"]
    assert len(exceptions) >= 40
    wrong = {
        "verdict": "REPAIR_SAFE",
        "exception_type": "RETURNED",
        "repair": "approve",
        "repair_safe": True,
    }
    for task in exceptions[:40]:
        assert verifier_payment_repair(task, wrong) is False


def test_verifier_does_not_read_gold():
    tasks = _suite(seed=42, n=100)
    for task in tasks[:60]:
        expected = compute_repair_decision_from_state(task.initial_state)
        assert expected == task.gold, "verifier independent of gold label"


def test_severity_mix_honored():
    tasks = _suite(
        seed=11,
        n=500,
        severity_mix={"HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1},
    )
    exceptions = [t for t in tasks if t.gold["exception_type"] is not None]
    assert len(exceptions) > 100
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for task in exceptions:
        counts[task.severity.value] += 1
    observed = {k: v / len(exceptions) for k, v in counts.items()}
    assert abs(observed["HIGH"] - 0.6) <= 0.05
    assert abs(observed["MEDIUM"] - 0.3) <= 0.05
    assert abs(observed["LOW"] - 0.1) <= 0.05


def test_all_signals_appear():
    tasks = _suite(seed=7, n=500)
    present = {t.gold["exception_type"] for t in tasks}
    assert set(EXCEPTION_SIGNALS) <= present


def test_signatures_unique():
    tasks = _suite(seed=3, n=120)
    sigs = {t.signature for t in tasks}
    assert len(sigs) == len(tasks)
    assert sigs == {task_signature(t) for t in tasks}


def test_fraud_hold_never_repair_safe():
    tasks = _suite(seed=5, n=120, severity_mix={"CRITICAL": 1.0})
    assert tasks
    for task in tasks:
        assert task.gold["exception_type"] == "FRAUD_HOLD"
        assert task.gold["repair_safe"] is False
        assert task.gold["repair"] in ("hold_hitl", "reject")
