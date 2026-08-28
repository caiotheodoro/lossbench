"""P3.3 fine-tune scaffolding + P3.6 full-run tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_export_training_data_contamination_free(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.finetune.export_training_data",
            "--out",
            str(tmp_path / "train.jsonl"),
            "--eval-out",
            str(tmp_path / "eval.jsonl"),
            "--train-n",
            "120",
            "--eval-n",
            "60",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    cert = json.loads(result.stdout)
    assert cert["contamination"]["overlap"] == 0.0
    assert cert["train_tasks"] == 120
    assert cert["eval_tasks"] == 60
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()


def test_train_mlx_dry_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.finetune.export_training_data",
            "--out",
            str(tmp_path / "train.jsonl"),
            "--eval-out",
            str(tmp_path / "eval.jsonl"),
            "--train-n",
            "60",
            "--eval-n",
            "30",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0
    plan = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.finetune.train_mlx",
            "--data",
            str(tmp_path / "train.jsonl"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert plan.returncode == 0, plan.stderr
    plan_line = next(line for line in plan.stdout.splitlines() if line.startswith("{"))
    parsed = json.loads(plan_line)
    assert parsed["config"]["steps"] == 740
    assert parsed["n_examples"] == 60


def test_train_mlx_missing_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.finetune.train_mlx",
            "--data",
            str(tmp_path / "nope.jsonl"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "training data not found" in result.stderr


def test_full_run_produces_artifacts(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.full_run",
            "--out",
            str(tmp_path / "artifacts"),
            "--seed",
            "7",
            "--n-tasks",
            "90",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr

    run_dirs = [p for p in (tmp_path / "artifacts").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]
    assert run_dir.name.startswith("2")
    assert "-stub-seed7-steps" in run_dir.name

    leaderboard = json.loads((run_dir / "leaderboard.json").read_text())
    assert leaderboard["runner"] == "stub"
    assert len(leaderboard["models"]) == 4
    for row in leaderboard["models"]:
        assert row["pass_at_1"] == 1.0
        assert row["severity_weighted_loss"] == 0.0
        assert "ece" in row

    report = (run_dir / "report.md").read_text()
    assert "| Model | Loss |" in report

    cert = json.loads((run_dir / "contamination_certificate.json").read_text())
    assert cert["valid"] is True
    assert cert["runner"] == "stub"

    runconfig = json.loads((run_dir / "runconfig.json").read_text())
    assert runconfig["runner"] == "stub"
    assert runconfig["seed"] == 7
    assert runconfig["model_ids"] == leaderboard_model_ids(leaderboard)
    assert runconfig["git_revision"]

    cards = sorted((run_dir / "model_cards").glob("*.md"))
    assert len(cards) == 4
    for card in cards:
        assert card.read_text().startswith("## ⚠️ STUB PIPELINE SMOKE OUTPUT")
    assert (run_dir / "workload.duckdb").exists()


def leaderboard_model_ids(leaderboard: dict) -> list[str]:
    return [row["model_id"] for row in leaderboard["models"]]
