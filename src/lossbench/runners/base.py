"""Core runner contracts shared by all model runner implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelRunner(Protocol):
    """A model backend that turns a prompt into a priced response."""

    name: str

    def decide(self, prompt: str, **params) -> RunnerResult:
        """Run one prompt through the model and return the priced result."""
        ...


@dataclass
class RunnerResult:
    """A single model call outcome with token and cost accounting."""

    text: str
    model_id: str
    latency_ms: float
    token_usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def compute_cost(
    token_usage: dict[str, int],
    cost_per_1k_in: float,
    cost_per_1k_out: float,
) -> float:
    """Dollar cost of a call from token counts and per-1k-token rates."""
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    in_cost = (prompt_tokens / 1000.0) * cost_per_1k_in
    out_cost = (completion_tokens / 1000.0) * cost_per_1k_out
    return in_cost + out_cost
