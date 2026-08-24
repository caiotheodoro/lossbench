"""Retry wrapper for transient gateway failures.

A benchmark run is thousands of sequential calls over hours. Without this a
single upstream timeout ends the run and discards every result collected so
far. Retries cover the transient case; a call that still fails is raised so
the harness can record it as an error rather than as a model mistake.
"""

from __future__ import annotations

import time

from lossbench.runners.base import ModelRunner, RunnerResult

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0


class RetryingRunner:
    """Retry a runner on any exception, with linear backoff between attempts."""

    def __init__(
        self,
        runner: ModelRunner,
        attempts: int = DEFAULT_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self._runner = runner
        self._attempts = attempts
        self._backoff = backoff_seconds

    @property
    def name(self) -> str:
        return self._runner.name

    def decide(self, prompt: str, **params) -> RunnerResult:
        """Call the wrapped runner, retrying transient failures.

        Re-raises the last exception once the attempt budget is spent, so a
        genuinely unreachable endpoint stays visible instead of turning into
        a silent zero.
        """
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                return self._runner.decide(prompt, **params)
            except Exception as exc:  # noqa: BLE001 - any provider error is retryable
                last = exc
                if attempt + 1 < self._attempts and self._backoff > 0:
                    time.sleep(self._backoff * (attempt + 1))
        raise last  # type: ignore[misc]


__all__ = ["DEFAULT_ATTEMPTS", "DEFAULT_BACKOFF_SECONDS", "RetryingRunner"]
