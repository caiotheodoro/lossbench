from lossbench.contamination import (
    leak_fraction_detected,
    monitor_report,
    signature_overlap,
    task_signature,
)
from lossbench.schema import Severity, Task

TRAIN_SIZE = 20
EVAL_SIZE = 20

_CURRENCIES = ("USD", "EUR", "BRL")
_BENEFICIARIES = ("Alpha", "Beta", "Gamma", "Delta", "Omega")
_SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def _make_task(i: int, id_prefix: str = "task") -> Task:
    amount = 1000.0 + i * 137.5
    currency = _CURRENCIES[i % len(_CURRENCIES)]
    beneficiary = _BENEFICIARIES[i % len(_BENEFICIARIES)]
    reference = f"REF-{i:04d}"
    value_date = f"2026-{1 + i % 12:02d}-{1 + (i * 7) % 28:02d}"
    prompt = (
        f"Reconcile payment {reference} of amount {amount} {currency} "
        f"to {beneficiary} on {value_date}."
    )
    return Task(
        id=f"{id_prefix}-{i:04d}",
        domain="reconciliation",
        prompt=prompt,
        initial_state={
            "reference": reference,
            "amount": amount,
            "currency": currency,
            "beneficiary": beneficiary,
            "value_date": value_date,
        },
        available_tools=["lookup", "verify"],
        policy_id="recon-v1",
        gold={"status": "match", "reference": reference},
        severity=Severity(_SEVERITIES[i % 4]),
        verifier="verifier_reconciliation",
        cost_model_ref="flat",
        difficulty=0.1 + (i % 10) / 10.0,
        seed=i * 7919,
    )


def _meta_only_copy(task: Task) -> Task:
    return task.model_copy(update={"id": f"{task.id}-meta", "seed": task.seed + 1})


def test_zero_overlap_clean_sets():
    train = [_make_task(i) for i in range(TRAIN_SIZE)]
    eval_set = [_make_task(TRAIN_SIZE + i) for i in range(EVAL_SIZE)]
    assert signature_overlap(train, eval_set) == 0.0
    report = monitor_report(train, eval_set)
    assert report["overlap"] == 0.0
    assert report["false_fire"] == 1.0
    assert report["detection"] == 1.0


def test_full_detection_at_any_leak_fraction():
    eval_set = [_make_task(TRAIN_SIZE + i) for i in range(EVAL_SIZE)]
    for fraction in (0.05, 0.2, 0.5):
        n_leaked = int(EVAL_SIZE * fraction)
        leaked_sources = eval_set[:n_leaked]
        leaked = [_meta_only_copy(t) for t in leaked_sources]
        train = [_make_task(i) for i in range(TRAIN_SIZE)]
        train += [_meta_only_copy(t) for t in leaked_sources]
        report = monitor_report(train, eval_set, leaked)
        assert report["detection"] >= 1.0
        assert report["overlap"] > 0.0
        assert leak_fraction_detected(train, eval_set, leaked) == 1.0


def test_metadata_ignored():
    for i in range(40):
        task = _make_task(i)
        clone = _meta_only_copy(task)
        assert task.id != clone.id
        assert task_signature(task) == task_signature(clone)


def test_identical_content_detected():
    train = [_make_task(i) for i in range(TRAIN_SIZE)]
    eval_set = [_meta_only_copy(t) for t in train]
    assert signature_overlap(train, eval_set) == 1.0
    report = monitor_report(train, eval_set)
    assert report["overlap"] == 1.0
    assert report["false_fire"] == 0.0


def test_empty_train():
    tasks = [_make_task(i) for i in range(10)]
    assert signature_overlap([], tasks) == 0.0
    assert signature_overlap([], []) == 0.0
    report = monitor_report([], tasks)
    assert report["overlap"] == 0.0
    assert report["false_fire"] == 1.0
    assert report["detection"] == 1.0


def test_signature_stability():
    task = _make_task(7)
    assert task_signature(task) == task_signature(task.model_copy(deep=True))
    sigs = {task_signature(_make_task(i)) for i in range(40)}
    assert len(sigs) == 40
