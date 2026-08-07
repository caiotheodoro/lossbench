"""Model runners: stub and OpenAI-compatible backends with cost tracking."""

from lossbench.runners.base import ModelRunner, RunnerResult
from lossbench.runners.baselines import BASELINE_MODELS
from lossbench.runners.register import make_runner
from lossbench.runners.stub import make_stub_runner

__all__ = [
    "BASELINE_MODELS",
    "ModelRunner",
    "RunnerResult",
    "make_runner",
    "make_stub_runner",
]
