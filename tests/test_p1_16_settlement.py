"""P1.16 settlement generator acceptance tests."""

from __future__ import annotations

from lossbench.generate.settlement import (
    SIGNALS,
    compute_verdict_from_state,
    generate_settlement_suite_internal,
    verifier_settlement,
)
from lossbench.generate.taxonomy import task_signature

_DIFFICULTY = (0.0, 1.0)


def _suite(seed, n_tasks, severity_mix=None, difficulty=_DIFFICULTY):
    return generate_settlement_suite_internal(
        seed, n_tasks, severity_mix, difficulty, verifier_settlement
    )


def test_same_seed_byte_identical():
    a = _suite(seed=7, n_tasks=50)
    b = _suite(seed=7, n_tasks=50)
    assert [t.model_dump_json() for t in a] == [t.model_dump_json() for t in b]


def test_verifier_agrees_with_gold_100pct():
    for seed in (7, 42, 777):
        tasks = _suite(seed=seed, n_tasks=100)
        for task in tasks:
            assert verifier_settlement(task, task.gold) is True


def test_verifier_does_not_read_gold():
    tasks = _suite(seed=42, n_tasks=100)
    for task in tasks:
        expected = compute_verdict_from_state(task.initial_state)
        assert expected == task.gold
        blinded = task.model_copy(update={"gold": {}})
        assert verifier_settlement(blinded, expected) is True


def test_severity_mix_honored():
    tasks = _suite(
        seed=11,
        n_tasks=400,
        severity_mix={"HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1},
    )
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for task in tasks:
        counts[task.severity.value] += 1
    observed = {k: v / len(tasks) for k, v in counts.items()}
    assert abs(observed["HIGH"] - 0.6) <= 0.05
    assert abs(observed["MEDIUM"] - 0.3) <= 0.05
    assert abs(observed["LOW"] - 0.1) <= 0.05


def test_all_signals_appear():
    tasks = _suite(seed=7, n_tasks=500)
    seen = {t.initial_state["settlement_signal"]["signal"] for t in tasks}
    assert set(SIGNALS) <= seen


def test_signatures_unique():
    tasks = _suite(seed=3, n_tasks=120)
    sigs = {t.signature for t in tasks}
    assert len(sigs) == len(tasks)
    assert sigs == {task_signature(t) for t in tasks}


def test_herstatt_always_hitl():
    tasks = _suite(seed=7, n_tasks=500)
    herstatt = [
        t for t in tasks if t.initial_state["settlement_signal"]["signal"] == "HERSTATT_EXPOSURE"
    ]
    assert herstatt
    for task in herstatt:
        assert task.gold["verdict"] == "HITL"
        assert task.gold["exposure_class"] == "CRITICAL"
        assert task.severity.value == "CRITICAL"


def test_verifier_rejects_wrong_outcome():
    tasks = _suite(seed=7, n_tasks=300)
    exceptions = [t for t in tasks if t.gold["exception_type"] is not None]
    assert exceptions
    wrong = {"verdict": "MATCH", "exception_type": None, "exposure_class": "LOW"}
    for task in exceptions[:40]:
        assert verifier_settlement(task, wrong) is False
