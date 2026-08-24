"""Spend ceiling: a real run must never make uncapped paid calls."""

from __future__ import annotations

import pytest

from lossbench.runners.base import RunnerResult
from lossbench.runners.budget import BudgetedRunner, BudgetExceeded, BudgetTracker


class _CostingRunner:
    """Runner that bills a fixed amount per call."""

    def __init__(self, name: str, cost: float) -> None:
        self.name = name
        self._cost = cost
        self.calls = 0

    def decide(self, prompt: str, **params) -> RunnerResult:
        self.calls += 1
        return RunnerResult(
            text='{"verdict": "MATCH"}',
            model_id=self.name,
            latency_ms=1.0,
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            cost=self._cost,
            raw={},
        )


def test_calls_pass_through_under_the_ceiling():
    inner = _CostingRunner("m", 1.0)
    runner = BudgetedRunner(inner, BudgetTracker(10.0))
    for _ in range(5):
        runner.decide("p")
    assert inner.calls == 5


def test_aborts_once_the_ceiling_is_crossed():
    inner = _CostingRunner("m", 1.0)
    tracker = BudgetTracker(2.5)
    runner = BudgetedRunner(inner, tracker)
    runner.decide("p")
    runner.decide("p")
    with pytest.raises(BudgetExceeded, match="2.50"):
        runner.decide("p")
    assert inner.calls == 2, "no call may be issued after the ceiling is crossed"


def test_tracker_is_shared_across_runners():
    """One ceiling covers the whole run, not one model each."""
    tracker = BudgetTracker(3.0)
    first = BudgetedRunner(_CostingRunner("a", 1.0), tracker)
    second = BudgetedRunner(_CostingRunner("b", 1.0), tracker)
    first.decide("p")
    first.decide("p")
    second.decide("p")
    with pytest.raises(BudgetExceeded):
        second.decide("p")
    assert tracker.spent == pytest.approx(3.0)


def test_zero_ceiling_means_unlimited():
    inner = _CostingRunner("m", 100.0)
    runner = BudgetedRunner(inner, BudgetTracker(0.0))
    for _ in range(3):
        runner.decide("p")
    assert inner.calls == 3


def test_name_is_forwarded():
    assert BudgetedRunner(_CostingRunner("m", 0.0), BudgetTracker(1.0)).name == "m"
