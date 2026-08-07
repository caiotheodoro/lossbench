"""Deterministic, network-free model runner for tests and dry runs."""

from __future__ import annotations

import time

from lossbench.runners.base import ModelRunner, RunnerResult, compute_cost

DEFAULT_TEXT = "stub default"


class StubRunner:
    """Returns canned responses for exact prompt matches, else a default text."""

    def __init__(
        self,
        name: str,
        responses: dict[str, str],
        *,
        default_text: str = DEFAULT_TEXT,
        model_id: str | None = None,
    ) -> None:
        self.name = name
        self._responses = dict(responses)
        self._default_text = default_text
        self._model_id = model_id or name
        self._cost_per_1k_in = 0.0
        self._cost_per_1k_out = 0.0

    def decide(self, prompt: str, **params) -> RunnerResult:
        """Return the canned response for an exact match, else the default text.

        Lookup order: `task_id` in params first (allows per-task canned
        responses when prompts are shared), then exact prompt match.
        """
        start = time.perf_counter()
        task_id = params.get("task_id")
        text = self._responses.get(task_id, self._responses.get(prompt, self._default_text))
        latency_ms = (time.perf_counter() - start) * 1000.0
        token_usage = {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
        }
        cost = compute_cost(token_usage, self._cost_per_1k_in, self._cost_per_1k_out)
        return RunnerResult(
            text=text,
            model_id=self._model_id,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost=cost,
            raw={},
        )


def make_stub_runner(name: str, responses: dict[str, str]) -> ModelRunner:
    """Build a deterministic stub runner with zero cost and tiny latency."""
    return StubRunner(name=name, responses=responses)
