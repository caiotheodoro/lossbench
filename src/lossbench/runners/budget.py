"""Hard spend ceiling for paid runs.

A full run is `domains x n_tasks x trials x max_steps x models` calls, and
a failed verify triggers a retry, so the call count is bounded by the
retry budget rather than by the task count. Wrap paid runners in
`BudgetedRunner` and share one `BudgetTracker` across every model so the
ceiling covers the run, not each model separately.
"""

from __future__ import annotations

from lossbench.runners.base import ModelRunner, RunnerResult


class BudgetExceeded(RuntimeError):
    """Raised when cumulative spend crosses the configured ceiling."""


class BudgetTracker:
    """Cumulative spend against a ceiling. A ceiling of 0.0 means unlimited.

    The ceiling is enforced *before* a call goes out, using the previous
    call's cost as the estimate for the next one. Checking only after the
    fact would let the run keep issuing paid calls past the limit, which
    is the failure this class exists to prevent.
    """

    def __init__(self, ceiling_usd: float) -> None:
        self.ceiling_usd = ceiling_usd
        self.spent = 0.0
        self._last_cost = 0.0

    def check(self) -> None:
        """Raise if the next call is likely to cross the ceiling."""
        if self.ceiling_usd <= 0.0:
            return
        if self.spent + self._last_cost > self.ceiling_usd:
            raise BudgetExceeded(
                f"spend ceiling reached: ${self.spent:.2f} spent of "
                f"${self.ceiling_usd:.2f}, next call ~${self._last_cost:.2f}"
            )

    def charge(self, cost: float) -> None:
        """Record what a completed call actually cost."""
        self.spent += cost
        self._last_cost = cost


class BudgetedRunner:
    """Runner wrapper that stops the run once the shared ceiling is crossed."""

    def __init__(self, runner: ModelRunner, tracker: BudgetTracker) -> None:
        self._runner = runner
        self._tracker = tracker

    @property
    def name(self) -> str:
        return self._runner.name

    def decide(self, prompt: str, **params) -> RunnerResult:
        """Refuse if the ceiling is in reach, else call and charge the tracker."""
        self._tracker.check()
        result = self._runner.decide(prompt, **params)
        self._tracker.charge(result.cost)
        return result


__all__ = ["BudgetExceeded", "BudgetTracker", "BudgetedRunner"]
