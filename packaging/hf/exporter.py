"""HF Community Evals packaging for LossBench (P1.18).

Builds the artifacts that register LossBench as a Hugging Face dataset with
Community Evals: the eval.yaml registration manifest, the dataset card, and
the canonical JSONL task export for training splits.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

from lossbench.schema import Task

DEFAULT_LICENSE = "cc-by-4.0"

_REQUIRED_TOP_KEYS = ("id", "description", "dataset", "task_types", "metric")

_TEMPLATE_PATH = Path(__file__).resolve().parent / "dataset_card_template.md"


def build_eval_yaml(
    benchmark_id: str,
    description: str,
    task_types: list[str],
    metric: str,
    dataset_repo: str,
    version: str = "0.1.0",
) -> dict[str, Any]:
    """Return the eval.yaml dict per HF Community Evals convention:
    {"id", "description", "version", "dataset": {"path", "revision"},
     "task_types", "metric": {"name", "higher_is_better": bool},
     "license", "paper": {"title", "url"} | None}.
    """
    return {
        "id": benchmark_id,
        "description": description,
        "version": version,
        "dataset": {"path": dataset_repo, "revision": "main"},
        "task_types": task_types,
        "metric": {"name": metric, "higher_is_better": False},
        "license": DEFAULT_LICENSE,
        "paper": None,
    }


def build_dataset_card(
    benchmark_id: str,
    description: str,
    license_name: str,
    task_count: int,
    domains: list[str],
    severity_taxonomy: list[str],
    cost_model_ids: list[str],
    contamination_policy: str,
    reproducibility_notes: str,
    contact: str,
    results_table: str = "No model results published for this revision yet.",
    honest_limits: str = "- synthetic tasks; severity costs are contested inputs",
) -> str:
    """Render the dataset card markdown from the bundled template.

    Sections: Overview, Tasks, License, Severity taxonomy, Cost models,
    Contamination policy, Reproducibility, Contact.
    """
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        benchmark_id=benchmark_id,
        description=description,
        license_name=license_name,
        task_count=str(task_count),
        domains=", ".join(domains),
        severity_bullets="\n".join(f"- {band}" for band in severity_taxonomy),
        cost_model_ids=", ".join(cost_model_ids),
        contamination_policy=contamination_policy,
        reproducibility_notes=reproducibility_notes,
        contact=contact,
        results_table=results_table,
        honest_limits=honest_limits,
    )


def validate_eval_yaml(yaml_dict: dict[str, Any]) -> None:
    """Raise ValueError on missing keys (id, description, dataset.path, task_types, metric.name)."""
    if not isinstance(yaml_dict, dict):
        raise ValueError("eval.yaml must be a mapping")
    for key in _REQUIRED_TOP_KEYS:
        if key not in yaml_dict:
            raise ValueError(f"eval.yaml missing required key '{key}'")
    dataset = yaml_dict["dataset"]
    if not isinstance(dataset, dict) or "path" not in dataset:
        raise ValueError("eval.yaml 'dataset' must be a dict containing 'path'")
    metric = yaml_dict["metric"]
    if not isinstance(metric, dict) or "name" not in metric:
        raise ValueError("eval.yaml 'metric' must be a dict containing 'name'")


def tasks_to_jsonl(tasks: list[Task], path: str) -> int:
    """Serialize tasks to canonical JSONL (model_dump_json), return count.

    The signature field (eval-set identity hash) is never written, so
    training exports cannot leak eval-set signatures.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(task.model_dump_json(exclude={"signature"}) + "\n")
    return len(tasks)
