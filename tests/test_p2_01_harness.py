import json

import pytest

from lossbench.cache.store import ResponseCache
from lossbench.eval.harness import EvalHarness, domain_verifier, summarize_suite
from lossbench.generate import generate_suite
from lossbench.runners import RunnerResult, make_stub_runner
from lossbench.schema import DecisionKind, Severity, Task


def _gold_text(task):
    return json.dumps(task.gold)


def test_run_task_success():
    tasks = generate_suite("reconciliation", seed=1, n_tasks=1)
    task = tasks[0]
    runner = make_stub_runner("stub-ok", {task.prompt: _gold_text(task)})
    result = EvalHarness(runner=runner).run_task(task, seed=0)
    assert result.success
    assert len(result.events) >= 1
    assert result.events[-1].decision == DecisionKind.ALLOW
    assert result.events[-1].calibrated_probability == 0.9


def test_run_task_failure():
    tasks = generate_suite("reconciliation", seed=2, n_tasks=1)
    task = tasks[0]
    runner = make_stub_runner("stub-bad", {task.prompt: "I am not a verdict"})
    result = EvalHarness(runner=runner).run_task(task, seed=0)
    assert not result.success
    assert result.events[-1].decision == DecisionKind.ALLOW
    assert result.events[-1].calibrated_probability == 0.1


class TaskGoldStub:
    def __init__(self, gold_by_task: dict[str, str]) -> None:
        self.name = "task-gold"
        self._gold_by_task = gold_by_task

    def decide(self, prompt: str, **params) -> RunnerResult:
        text = self._gold_by_task[str(params.get("task_id", ""))]
        return RunnerResult(
            text=text,
            model_id=self.name,
            latency_ms=0.0,
            token_usage={"prompt_tokens": 8, "completion_tokens": 4},
            cost=0.0,
        )


class PartialStub:
    def __init__(self, gold_by_task: dict[str, str], wrong_text: str) -> None:
        self.name = "partial"
        self._gold_by_task = gold_by_task
        self._wrong = wrong_text

    def decide(self, prompt: str, **params) -> RunnerResult:
        task_id = str(params.get("task_id", ""))
        idx = int(task_id.rpartition(":")[2])
        seed = int(params.get("seed", 0))
        ok = idx % 2 == 0 or seed % 3 != 1
        text = self._gold_by_task.get(task_id, self._wrong) if ok else self._wrong
        return RunnerResult(
            text=text,
            model_id=self.name,
            latency_ms=0.0,
            token_usage={"prompt_tokens": 8, "completion_tokens": 4},
            cost=0.0,
        )


def test_run_suite_passk_semantics():
    tasks = generate_suite("reconciliation", seed=1, n_tasks=10)
    gold_by_task = {t.id: _gold_text(t) for t in tasks}
    runner = PartialStub(gold_by_task, wrong_text="WRONG VERDICT")
    results = EvalHarness(runner=runner).run_suite(tasks, trials=3, seed=0)
    assert len(results) == 30
    by_task: dict[str, list[bool]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r.success)
    for task_id, outcomes in by_task.items():
        assert len(outcomes) == 3
        idx = int(task_id.rpartition(":")[2])
        if idx % 2 == 0:
            assert all(outcomes)
        else:
            assert not all(outcomes)
    summary = summarize_suite(results)
    assert summary["pass_at_1"] == 1.0
    assert summary["pass_at_k"] == 1.0
    assert summary["pass_k"] == 0.5


def test_summarize_suite_keys():
    tasks = generate_suite("reconciliation", seed=4, n_tasks=3)
    gold_by_task = {t.id: _gold_text(t) for t in tasks}
    runner = TaskGoldStub(gold_by_task)
    results = EvalHarness(runner=runner).run_suite(tasks, trials=2, seed=0)
    summary = summarize_suite(results)
    assert set(summary) == {
        "tasks",
        "trials",
        "pass_at_1",
        "pass_at_k",
        "pass_k",
        "total_cost",
        "mean_duration_ms",
        "false_success_rate",
        "parse_rate",
        "error_rate",
    }
    assert summary["tasks"] == 3
    assert summary["trials"] == 2
    assert summary["pass_at_1"] == 1.0
    assert summary["pass_at_k"] == 1.0
    assert summary["pass_k"] == 1.0
    assert summary["total_cost"] == 0.0
    assert summary["mean_duration_ms"] > 0.0
    assert 0.0 <= summary["false_success_rate"] <= 1.0


class CountingStub:
    def __init__(self, name: str, gold_by_task: dict[str, str]) -> None:
        self.name = name
        self._gold_by_task = gold_by_task
        self.calls = 0

    def decide(self, prompt: str, **params) -> RunnerResult:
        self.calls += 1
        text = self._gold_by_task[str(params.get("task_id", ""))]
        return RunnerResult(
            text=text,
            model_id=self.name,
            latency_ms=0.0,
            token_usage={"prompt_tokens": 8, "completion_tokens": 4},
            cost=0.0,
        )


def test_cache_reuse():
    tasks = generate_suite("reconciliation", seed=5, n_tasks=1)
    task = tasks[0]
    runner = CountingStub("counting", {task.id: _gold_text(task)})
    harness = EvalHarness(runner=runner, cache=ResponseCache())
    first = harness.run_task(task, seed=0)
    assert first.success
    assert runner.calls == 1
    second = harness.run_task(task, seed=0)
    assert second.success
    assert runner.calls == 1


def test_domain_verifier_dispatch():
    tasks = generate_suite("reconciliation", seed=1, n_tasks=5)
    for task in tasks:
        assert domain_verifier(task, task.gold)
        assert not domain_verifier(task, {"verdict": "WRONG", "exception_type": None})
    bad = Task(
        id="x",
        domain="reconciliation",
        prompt="p",
        policy_id="p",
        gold={},
        severity=Severity.LOW,
        verifier="verifier_unknown",
        cost_model_ref="c",
        seed=0,
    )
    with pytest.raises(ValueError):
        domain_verifier(bad, {})


def test_deterministic_with_stub():
    tasks = generate_suite("reconciliation", seed=7, n_tasks=3)
    gold_by_task = {t.id: _gold_text(t) for t in tasks}
    first = EvalHarness(runner=TaskGoldStub(gold_by_task)).run_suite(
        tasks, trials=3, seed=1
    )
    second = EvalHarness(runner=TaskGoldStub(gold_by_task)).run_suite(
        tasks, trials=3, seed=1
    )
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.task_id == b.task_id
        assert a.success == b.success
        assert [e.model_dump() for e in a.events] == [e.model_dump() for e in b.events]


def test_max_steps_bounded():
    tasks = generate_suite("reconciliation", seed=8, n_tasks=1)
    task = tasks[0]
    runner = make_stub_runner("stub-never", {})
    result = EvalHarness(runner=runner, max_steps=4).run_task(task, seed=0)
    assert not result.success
    assert len(result.events) <= 4
    assert len(result.events) == 4
    assert result.events[-1].decision == DecisionKind.ALLOW
    assert all(e.decision == DecisionKind.VERIFY for e in result.events[:-1])
