"""Hermetic tests for the P1.7 model runners package. No network access."""

from dataclasses import is_dataclass

import pytest

import lossbench.runners.openai_compat as openai_compat
from lossbench.runners import (
    BASELINE_MODELS,
    ModelRunner,
    RunnerResult,
    make_runner,
    make_stub_runner,
)
from lossbench.runners.stub import compute_cost

EXPECTED_BASELINE_IDS = {
    "reconforge-1.7b",
    "qwen3.8-27b",
    "nemotron-3.5-lightning",
    "deepseek-v4-flash",
    "qwen3.8-2.4t-a95b",
}


def test_stub_exact_match():
    runner = make_stub_runner("stub", {"hello": "hi there"})
    exact = runner.decide("hello")
    default = runner.decide("unseen prompt")
    assert isinstance(runner, ModelRunner)
    assert exact.text == "hi there"
    assert default.text == "stub default"
    assert exact.cost == 0.0
    assert exact.latency_ms >= 0.0
    assert {"prompt_tokens", "completion_tokens"} <= set(exact.token_usage)
    assert exact.model_id == "stub"


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="[Uu]nknown"):
        make_runner("does-not-exist")


def test_openai_compat_missing_key_raises(monkeypatch):
    monkeypatch.delenv("LOSSBENCH_TEST_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LOSSBENCH_TEST_MISSING_KEY"):
        make_runner(
            "openai_compat",
            model_id="some-model",
            api_key_env="LOSSBENCH_TEST_MISSING_KEY",
        )


def test_cost_calculation():
    token_usage = {"prompt_tokens": 1000, "completion_tokens": 2000}
    cost = compute_cost(token_usage, cost_per_1k_in=0.5, cost_per_1k_out=1.0)
    assert cost == pytest.approx(2.5)
    result = RunnerResult(
        text="x",
        model_id="m",
        latency_ms=1.0,
        token_usage=token_usage,
        cost=cost,
    )
    assert result.cost == pytest.approx(2.5)


def test_baseline_registry():
    assert set(BASELINE_MODELS) == EXPECTED_BASELINE_IDS
    for config in BASELINE_MODELS.values():
        assert {"model_id", "cost_per_1k_in", "cost_per_1k_out"} <= set(config)


class _FakeOpenAI:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeOpenAI.instances.append(self)


def test_openai_compat_url_construction(monkeypatch):
    monkeypatch.setenv("LOSSBENCH_TEST_API_KEY", "sk-test-123")
    monkeypatch.setattr(openai_compat.openai, "OpenAI", _FakeOpenAI)
    make_runner(
        "openai_compat",
        model_id="some-model",
        api_key_env="LOSSBENCH_TEST_API_KEY",
        base_url="http://localhost:8000/v1",
    )
    client = _FakeOpenAI.instances[-1]
    assert client.kwargs["base_url"] == "http://localhost:8000/v1"
    assert client.kwargs["api_key"] == "sk-test-123"


def test_runner_result_is_dataclass():
    assert is_dataclass(RunnerResult)
    result = RunnerResult(text="t", model_id="m", latency_ms=0.5)
    assert result.token_usage == {}
    assert result.raw == {}
