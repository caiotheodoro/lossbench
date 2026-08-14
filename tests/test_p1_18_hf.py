import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from lossbench.schema import Severity, Task

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPORTER_PATH = _REPO_ROOT / "packaging" / "hf" / "exporter.py"
_EVAL_YAML_PATH = _REPO_ROOT / "packaging" / "hf" / "eval.yaml"

_CARD_SECTIONS = [
    "Overview",
    "Tasks",
    "License",
    "Severity taxonomy",
    "Cost models",
    "Contamination policy",
    "Reproducibility",
    "Contact",
]


def _load_exporter():
    spec = importlib.util.spec_from_file_location("lossbench_hf_exporter", _EXPORTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_exporter()


def _task(i: int) -> Task:
    return Task(
        id=f"task-{i:03d}",
        domain="reconciliation",
        prompt=f"Reconcile statement {i}",
        initial_state={"balance": 100.0 + i},
        available_tools=["ledger", "apply"],
        policy_id="pol-test",
        gold={"tool": "commit", "entry": i},
        severity=Severity.LOW if i % 2 == 0 else Severity.HIGH,
        verifier="lossbench.generate.reconciliation.verifier_reconciliation",
        cost_model_ref="reconciliation",
        difficulty=0.5,
        seed=42,
    )


def _eval_doc() -> dict:
    return exporter.build_eval_yaml(
        benchmark_id="lossbench",
        description="Expected-loss evaluation for agentic back-office systems",
        task_types=["text-generation"],
        metric="severity_weighted_loss",
        dataset_repo="LossBench/lossbench",
    )


def _card() -> str:
    return exporter.build_dataset_card(
        benchmark_id="lossbench",
        description="Expected-loss evaluation for agentic back-office systems",
        license_name="cc-by-4.0",
        task_count=1000,
        domains=["reconciliation", "payment_repair", "settlement"],
        severity_taxonomy=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        cost_model_ids=["flat", "reconciliation", "principal_risk", "review_heavy"],
        contamination_policy=(
            "All generated tasks are screened against public corpora; "
            "eval-set signatures are never published."
        ),
        reproducibility_notes="Identical seeds reproduce byte-identical suites and results.",
        contact="LossBench team (lossbench@example.com)",
    )


def test_eval_yaml_shape():
    doc = _eval_doc()
    assert list(doc) == [
        "id",
        "description",
        "version",
        "dataset",
        "task_types",
        "metric",
        "license",
        "paper",
    ]
    assert doc["dataset"] == {"path": "LossBench/lossbench", "revision": "main"}
    assert doc["task_types"] == ["text-generation"]
    assert doc["metric"]["name"] == "severity_weighted_loss"
    assert doc["metric"]["higher_is_better"] is False
    assert doc["license"] == "cc-by-4.0"
    assert doc["paper"] is None


def test_validate_eval_yaml_accepts_valid():
    exporter.validate_eval_yaml(_eval_doc())
    shipped = yaml.safe_load(_EVAL_YAML_PATH.read_text(encoding="utf-8"))
    exporter.validate_eval_yaml(shipped)
    assert shipped["metric"]["name"] == "severity_weighted_loss"
    assert shipped["metric"]["higher_is_better"] is False


@pytest.mark.parametrize(
    "remove",
    ["id", "description", "task_types", ("dataset", "path"), ("metric", "name")],
)
def test_validate_eval_yaml_rejects_missing(remove):
    doc = _eval_doc()
    if isinstance(remove, tuple):
        doc[remove[0]].pop(remove[1])
    else:
        doc.pop(remove)
    with pytest.raises(ValueError):
        exporter.validate_eval_yaml(doc)


def test_dataset_card_sections():
    card = _card()
    for section in _CARD_SECTIONS:
        assert f"## {section}" in card
    assert "pluggable" in card


def test_tasks_to_jsonl_roundtrip(tmp_path):
    tasks = [_task(i) for i in range(10)]
    out = tmp_path / "tasks.jsonl"
    count = exporter.tasks_to_jsonl(tasks, str(out))
    assert count == 10
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    reloaded = [Task.model_validate_json(line) for line in lines]
    assert reloaded == tasks


def test_tasks_to_jsonl_canonical(tmp_path):
    tasks = [_task(i) for i in range(10)]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    exporter.tasks_to_jsonl(tasks, str(first))
    exporter.tasks_to_jsonl(tasks, str(second))
    assert first.read_bytes() == second.read_bytes()


def test_tasks_to_jsonl_empty(tmp_path):
    out = tmp_path / "empty.jsonl"
    assert exporter.tasks_to_jsonl([], str(out)) == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_card_deterministic():
    first = _card()
    second = _card()
    assert first == second
