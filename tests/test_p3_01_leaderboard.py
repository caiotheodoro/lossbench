"""P3.1 leaderboard Space tests.

Modules under spaces/ are not a package; load them via importlib like the
packaging/hf tests do.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import gradio as gr
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEADERBOARD_PATH = _REPO_ROOT / "spaces" / "leaderboard" / "leaderboard.py"
_APP_PATH = _REPO_ROOT / "spaces" / "leaderboard" / "app.py"
_SAMPLE_PATH = _REPO_ROOT / "spaces" / "leaderboard" / "sample_leaderboard.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


leaderboard = _load_module("leaderboard", _LEADERBOARD_PATH)
app = _load_module("lossbench_leaderboard_app", _APP_PATH)

LeaderboardRow = leaderboard.LeaderboardRow


def _write(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_load_leaderboard_sorted(tmp_path):
    path = _write(
        tmp_path / "board.json",
        {
            "models": [
                {"model_id": "m-b", "loss": 3.0},
                {"model_id": "m-a", "loss": 1.0},
                {"model_id": "m-c", "loss": 2.0},
            ]
        },
    )
    rows = leaderboard.load_leaderboard(path)
    assert [row.model_id for row in rows] == ["m-a", "m-c", "m-b"]
    assert [row.severity_weighted_loss for row in rows] == [1.0, 2.0, 3.0]


def test_load_leaderboard_missing_optional(tmp_path):
    path = _write(tmp_path / "board.json", {"models": [{"model_id": "m", "loss": 0.5}]})
    row = leaderboard.load_leaderboard(path)[0]
    assert row.pass_k is None
    assert row.ece is None
    assert row.escalated is None
    assert row.total_cost is None
    assert row.model_id == "m"
    assert row.severity_weighted_loss == 0.5


def test_load_leaderboard_invalid_raises(tmp_path):
    path = _write(tmp_path / "board.json", {"models": [{"model_id": "m"}]})
    with pytest.raises(ValueError, match="loss"):
        leaderboard.load_leaderboard(path)
    path = _write(tmp_path / "board.json", {"models": [{"loss": 1.0}]})
    with pytest.raises(ValueError, match="model_id"):
        leaderboard.load_leaderboard(path)


def test_render_table_shape():
    rows = [
        LeaderboardRow("model-a", 0.4219, pass_k=0.87, ece=0.0312, escalated=14, total_cost=92.5),
        LeaderboardRow("model-b", 1.6),
    ]
    out = leaderboard.render_table(rows)
    lines = out.splitlines()
    assert lines[0] == "| Model | Loss | pass^k | ECE | Escalated | Cost |"
    assert "| model-a | 0.4219 | 0.8700 | 0.0312 | 14 | 92.5000 |" in lines
    assert "| model-b | 1.6000 | - | - | - | - |" in lines


def test_crossover_summary_with_crossings():
    sensitivities = {
        "model_a": [{"ratio": 1.0, "loss": 0.42}, {"ratio": 10.0, "loss": 4.02}],
        "model_b": [{"ratio": 1.0, "loss": 0.9}, {"ratio": 10.0, "loss": 3.6}],
    }
    out = leaderboard.crossover_summary(sensitivities)
    assert "model_a" in out
    assert "model_b" in out
    assert "ratio 10" in out


def test_crossover_summary_flat():
    sensitivities = {
        "a": [{"ratio": 1.0, "loss": 0.5}, {"ratio": 10.0, "loss": 1.0}],
        "b": [{"ratio": 1.0, "loss": 0.6}, {"ratio": 10.0, "loss": 1.2}],
    }
    assert leaderboard.crossover_summary(sensitivities) == "no crossovers"


def test_crossover_summary_none():
    assert leaderboard.crossover_summary(None) == "no sensitivities"


def test_demo_constructs():
    blocks = app.demo()
    assert isinstance(blocks, gr.Blocks)


def test_sample_fixture_loads():
    rows = leaderboard.load_leaderboard(_SAMPLE_PATH)
    assert len(rows) == 3
    assert [row.model_id for row in rows] == ["demo-model-a", "demo-model-b", "demo-model-c"]
    assert rows[2].pass_k is None
    assert rows[2].ece is None
    assert [row.severity_weighted_loss for row in rows] == sorted(
        row.severity_weighted_loss for row in rows
    )


def test_sample_fixture_is_obviously_synthetic():
    # No fabricated real model IDs, and a loud banner the UI renders as a warning.
    text = _SAMPLE_PATH.read_text(encoding="utf-8")
    for forbidden in ("reconforge", "qwen3-8b", "baseline-gpt-4o"):
        assert forbidden not in text
    assert leaderboard.load_banner(_SAMPLE_PATH) == "SYNTHETIC DEMO DATA"
    assert app._banner_md(str(_SAMPLE_PATH)).startswith("> ## ⚠")


def test_real_artifact_parses_end_to_end():
    # The real, committed run artifact must load through the Space loader.
    artifact = _REPO_ROOT / "artifacts" / "leaderboard.json"
    assert artifact.exists(), "artifacts/leaderboard.json must be committed"
    rows = leaderboard.load_leaderboard(artifact)
    assert rows, "real artifact should yield leaderboard rows"
    sensitivities, limits = leaderboard._load_extras(artifact)
    table = leaderboard.render_table(rows)
    assert table.startswith("| Model |")
    leaderboard.crossover_summary(sensitivities)
    leaderboard._format_honest_limits(limits)


def test_severity_weighted_loss_key_accepted(tmp_path):
    path = _write(
        tmp_path / "board.json",
        {"models": [{"model_id": "m", "severity_weighted_loss": 0.7}]},
    )
    row = leaderboard.load_leaderboard(path)[0]
    assert row.severity_weighted_loss == 0.7


def test_legacy_loss_alias_still_accepted(tmp_path):
    path = _write(tmp_path / "board.json", {"models": [{"model_id": "m", "loss": 0.3}]})
    assert leaderboard.load_leaderboard(path)[0].severity_weighted_loss == 0.3
