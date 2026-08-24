"""Transient gateway failures must not destroy a long run, or fake a result."""

from __future__ import annotations

import pytest

from lossbench.eval.harness import EvalHarness, summarize_suite
from lossbench.generate import generate_suite
from lossbench.runners.base import RunnerResult
from lossbench.runners.retry import RetryingRunner


class _FlakyRunner:
    """Fails the first `failures` calls, then answers."""

    def __init__(self, failures: int, exc: Exception | None = None) -> None:
        self.name = "flaky"
        self._left = failures
        self._exc = exc or TimeoutError("upstream timed out")
        self.calls = 0

    def decide(self, prompt: str, **params) -> RunnerResult:
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise self._exc
        return RunnerResult(
            text='{"verdict": "MATCH", "exception_type": null}',
            model_id=self.name,
            latency_ms=1.0,
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            cost=0.0,
            raw={},
        )


def test_retries_a_transient_failure():
    inner = _FlakyRunner(failures=2)
    runner = RetryingRunner(inner, attempts=3, backoff_seconds=0.0)
    assert runner.decide("p").text.startswith("{")
    assert inner.calls == 3


def test_gives_up_after_the_attempt_budget():
    inner = _FlakyRunner(failures=99)
    runner = RetryingRunner(inner, attempts=3, backoff_seconds=0.0)
    with pytest.raises(TimeoutError):
        runner.decide("p")
    assert inner.calls == 3


def test_name_is_forwarded():
    assert RetryingRunner(_FlakyRunner(0), attempts=1).name == "flaky"


class _AlwaysFailing:
    name = "dead"

    def decide(self, prompt: str, **params) -> RunnerResult:
        raise TimeoutError("upstream timed out")


def test_a_dead_runner_does_not_abort_the_suite():
    """One unreachable model must not throw away the whole run."""
    tasks = generate_suite("reconciliation", seed=777, n_tasks=3)
    results = EvalHarness(_AlwaysFailing(), max_steps=1).run_suite(tasks, trials=1, seed=0)
    assert len(results) == 3
    assert all(not r.success for r in results)


def test_runner_errors_are_reported_separately_from_model_mistakes():
    """A gateway failure is not evidence about the model; it must be visible."""
    tasks = generate_suite("reconciliation", seed=777, n_tasks=4)
    results = EvalHarness(_AlwaysFailing(), max_steps=1).run_suite(tasks, trials=1, seed=0)
    summary = summarize_suite(results)
    assert summary["error_rate"] == 1.0
    assert summary["parse_rate"] == 0.0


def test_clean_run_reports_no_errors():
    tasks = generate_suite("reconciliation", seed=777, n_tasks=3)
    runner = _FlakyRunner(failures=0)
    results = EvalHarness(runner, max_steps=1).run_suite(tasks, trials=1, seed=0)
    assert summarize_suite(results)["error_rate"] == 0.0
