"""Factory for model runners by name."""

from __future__ import annotations

from lossbench.runners.base import ModelRunner
from lossbench.runners.openai_compat import OpenAICompatRunner
from lossbench.runners.stub import make_stub_runner

RUNNER_KINDS = ("stub", "openai_compat")


def make_runner(name: str, **config) -> ModelRunner:
    """Build a runner by kind name; unknown kinds raise ValueError."""
    if name == "stub":
        return make_stub_runner(
            name=config.pop("name", "stub"),
            responses=config.pop("responses", {}),
        )
    if name == "openai_compat":
        return OpenAICompatRunner(**config)
    raise ValueError(f"Unknown runner {name!r}; expected one of: {', '.join(RUNNER_KINDS)}")
