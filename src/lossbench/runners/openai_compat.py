"""OpenAI-compatible chat completions runner with cost tracking."""

from __future__ import annotations

import os
import time

import openai

from lossbench.runners.base import RunnerResult, compute_cost

FORWARDED_PARAMS = ("reasoning_effort", "temperature", "top_p")


class OpenAICompatRunner:
    """Calls any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str,
        base_url: str | None = None,
        cost_per_1k_in: float = 0.0,
        cost_per_1k_out: float = 0.0,
        name: str | None = None,
        **params: object,
    ) -> None:
        self.name = name or model_id
        self._model_id = model_id
        self._cost_per_1k_in = cost_per_1k_in
        self._cost_per_1k_out = cost_per_1k_out
        self._params = params
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: environment variable {api_key_env!r} is not set")
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def decide(self, prompt: str, **params) -> RunnerResult:
        """Send the prompt to the endpoint and return the priced response."""
        forwarded = {k: v for k, v in {**self._params, **params}.items() if k in FORWARDED_PARAMS}
        messages = [{"role": "user", "content": prompt}]
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            timeout=60,
            **forwarded,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = response.choices[0].message.content
        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
        cost = compute_cost(token_usage, self._cost_per_1k_in, self._cost_per_1k_out)
        raw = {"text": text, "usage": token_usage}
        try:
            raw = response.model_dump()
        except AttributeError:
            pass
        return RunnerResult(
            text=text,
            model_id=self._model_id,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost=cost,
            raw=raw,
        )
