import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from lossbench.schema import Severity, Task

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPORTER_PATH = _REPO_ROOT / "packaging" / "hf" / "exporter.py"
_EVAL_YAML_PATH = _REPO_ROOT / "packaging" / "hf" / "eval.yaml"
_PUBLISH_PATH = _REPO_ROOT / "packaging" / "hf" / "publish.py"

_CARD_SECTIONS = [
    "Overview",
    "Tasks",
    "License",
    "Severity taxonomy",
    "Cost models",
    "Coverage",
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


def _load_publish():
    spec = importlib.util.spec_from_file_location("lossbench_hf_publish", _PUBLISH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


publish = _load_publish()


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


def test_card_has_yaml_front_matter():
    """Without front matter the Hub shows the dataset untagged and unlicensed."""
    card = _card()
    assert card.startswith("---\n")
    body = card.split("---\n", 2)
    assert len(body) == 3, "card must open with a closed YAML front-matter block"
    meta = yaml.safe_load(body[1])
    assert meta["license"] == "cc-by-4.0"
    assert "text-generation" in meta["task_categories"]
    assert meta["pretty_name"]
    assert meta["tags"]


def test_card_front_matter_declares_both_splits():
    meta = yaml.safe_load(_card().split("---\n", 2)[1])
    configs = meta["configs"]
    assert len(configs) == 1
    splits = {entry["split"]: entry["path"] for entry in configs[0]["data_files"]}
    assert splits == {"eval": "data/eval.jsonl", "train": "data/train.jsonl"}


def test_card_renders_every_placeholder():
    """string.Template.substitute leaves no $placeholder behind."""
    assert "$" not in _card()


def test_tasks_to_jsonl_excludes_the_signature(tmp_path):
    """Signature exclusion is the stated contamination guarantee."""
    import json as _json

    out = tmp_path / "tasks.jsonl"
    exporter.tasks_to_jsonl([_task(0)], str(out))
    record = _json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "signature" not in record


def _leaderboard(tmp_path, models):
    import json as _json

    path = tmp_path / "leaderboard.json"
    path.write_text(
        _json.dumps(
            {
                "runner": "openai_compat",
                "cost_model": "reconciliation",
                "generated_at": "2026-08-27T00:00:00+00:00",
                "models": models,
            }
        )
    )
    return path


def test_results_table_reads_false_success_applicable_flag(tmp_path):
    """_results_table must trust the flag full_run.py threaded into each row,
    not re-derive it by sniffing false_success_rate's type (issue #10 review)."""
    markdown, _ = publish._results_table(
        _leaderboard(
            tmp_path,
            [
                {
                    "model_id": "m",
                    "severity_weighted_loss": 0.1,
                    "pass_at_1": 0.9,
                    "pass_k": 0.9,
                    "false_success_rate": 0.25,
                    "false_success_applicable": True,
                }
            ],
        )
    )
    assert "| False-success" in markdown
    assert "0.250" in markdown


def test_results_table_omits_false_success_when_not_applicable_even_if_rate_is_numeric(tmp_path):
    """A row could carry a stale numeric false_success_rate alongside
    false_success_applicable=False (e.g. a self-verifying harness row that
    hasn't been fully migrated to null yet) -- the flag must win, not the type
    of the rate field, or a harness-scored 0.0 renders as a real measurement."""
    markdown, _ = publish._results_table(
        _leaderboard(
            tmp_path,
            [
                {
                    "model_id": "m",
                    "severity_weighted_loss": 0.1,
                    "pass_at_1": 0.9,
                    "pass_k": 0.9,
                    "false_success_rate": 0.0,
                    "false_success_applicable": False,
                }
            ],
        )
    )
    assert "| False-success" not in markdown
    assert "| 0.000" not in markdown
