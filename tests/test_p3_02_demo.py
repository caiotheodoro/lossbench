"""P3.2 demo Space acceptance tests: workload build, simulate_ui, Gradio UI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import gradio as gr

from lossbench.eval.harness import EvalHarness
from lossbench.generate import generate_suite
from lossbench.runners import make_stub_runner

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEMO_PATH = _REPO_ROOT / "spaces" / "demo" / "demo.py"
_EXPECTED_KEYS = {
    "before_loss",
    "after_loss",
    "before_review_load",
    "after_review_load",
    "n_events",
    "n_cases_changed",
    "markdown",
}


def _load_demo():
    spec = importlib.util.spec_from_file_location("lossbench_demo", _DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo_mod = _load_demo()


def test_build_workload_deterministic():
    events_a, ledger_a = demo_mod.build_workload(seed=7, n_tasks=40)
    events_b, ledger_b = demo_mod.build_workload(seed=7, n_tasks=40)
    assert [event.model_dump() for event in events_a] == [event.model_dump() for event in events_b]
    assert ledger_a.verify()["valid"]
    assert ledger_b.verify()["valid"]
    assert ledger_a.verify()["n_events"] == len(events_a)
    assert events_a


def test_simulate_ui_shapes():
    events, _ = demo_mod.build_workload(seed=7, n_tasks=40)
    result = demo_mod.simulate_ui(events, 0.5, 0.9)
    assert set(result) == _EXPECTED_KEYS
    assert result["n_events"] == len(events)
    assert "before" in result["markdown"]
    assert "after" in result["markdown"]


def test_simulate_ui_threshold_sensitivity():
    tasks = generate_suite("reconciliation", seed=7, n_tasks=50)
    responses = {
        task.id: json.dumps(task.gold, sort_keys=True)
        for task in tasks
        if task.gold.get("verdict") == "EXCEPTION"
    }
    harness = EvalHarness(runner=make_stub_runner("sensitivity-stub", responses))
    events = [
        event for result in harness.run_suite(tasks, trials=1, seed=0) for event in result.events
    ]
    probs = [event.calibrated_probability for event in events]
    assert any(p > 0.5 for p in probs)
    assert any(p < 0.5 for p in probs)
    strict = demo_mod.simulate_ui(events, 0.5, 1.0)
    lax = demo_mod.simulate_ui(events, 0.5, 0.0)
    assert strict["after_loss"] != lax["after_loss"]


def test_simulate_ui_empty_events():
    result = demo_mod.simulate_ui([], 0.5, 0.9)
    assert result["before_loss"] == 0.0
    assert result["after_loss"] == 0.0
    assert result["before_review_load"] == 0.0
    assert result["after_review_load"] == 0.0
    assert result["n_events"] == 0
    assert result["n_cases_changed"] == 0


def test_demo_constructs():
    blocks = demo_mod.demo()
    assert isinstance(blocks, gr.Blocks)


def test_deterministic():
    events, _ = demo_mod.build_workload(seed=7, n_tasks=40)
    first = demo_mod.simulate_ui(events, 0.4, 0.8, cost_model="flat")
    second = demo_mod.simulate_ui(events, 0.4, 0.8, cost_model="flat")
    assert first == second


def test_readme_exists():
    readme = _REPO_ROOT / "spaces" / "demo" / "README.md"
    assert readme.exists()
    assert "re-run last month" in readme.read_text()
