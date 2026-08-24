"""Calibrated probability must come from the model, not from its own score.

Before this, the harness set calibrated_probability to 0.9 when the answer
was right and 0.1 when it was wrong. That is a function of correctness, so
every model scored an identical ECE and no threshold sweep could ever change
a decision. Calibration is the centre of the expected-loss thesis, so it has
to be a measurement.
"""

from __future__ import annotations

import json

import pytest

from lossbench.eval.harness import EvalHarness
from lossbench.generate import DOMAINS, generate_suite
from lossbench.generate.prompt import OUTPUT_SCHEMAS
from lossbench.runners.base import RunnerResult


class _ScriptedRunner:
    """Replies with a fixed JSON body for every task."""

    def __init__(self, payload: dict) -> None:
        self.name = "scripted"
        self._payload = payload

    def decide(self, prompt: str, **params) -> RunnerResult:
        return RunnerResult(
            text=json.dumps(self._payload),
            model_id=self.name,
            latency_ms=1.0,
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            cost=0.0,
            raw={},
        )


@pytest.mark.parametrize("domain", DOMAINS)
def test_confidence_is_part_of_the_output_contract(domain):
    assert "confidence" in OUTPUT_SCHEMAS[domain]
    task = generate_suite(domain, seed=777, n_tasks=1)[0]
    assert '"confidence"' in task.prompt


def _run_once(payload: dict):
    task = generate_suite("reconciliation", seed=777, n_tasks=1)[0]
    results = EvalHarness(_ScriptedRunner(payload), max_steps=1).run_suite(
        [task], trials=1, seed=0
    )
    return results[0].events[-1]


def test_model_confidence_is_recorded():
    gold = generate_suite("reconciliation", seed=777, n_tasks=1)[0].gold
    event = _run_once({**gold, "confidence": 0.42})
    assert event.calibrated_probability == pytest.approx(0.42)
    assert event.risk_features["calibrated_p"] == pytest.approx(0.42)


def test_confidence_is_independent_of_correctness():
    """A confident wrong answer must record high confidence, not 0.1."""
    event = _run_once(
        {"verdict": "EXCEPTION", "exception_type": "AMOUNT_MISMATCH", "confidence": 0.95}
    )
    assert event.calibrated_probability == pytest.approx(0.95)


@pytest.mark.parametrize("bad", [None, "high", -0.5, 1.5, float("nan")])
def test_unusable_confidence_falls_back(bad):
    """A missing or out-of-range value must not become a fake measurement."""
    gold = generate_suite("reconciliation", seed=777, n_tasks=1)[0].gold
    event = _run_once({**gold, "confidence": bad})
    assert 0.0 <= event.calibrated_probability <= 1.0


def test_confidence_is_not_verified_against_gold():
    """Adding confidence to the contract must not change what counts as correct."""
    gold = generate_suite("reconciliation", seed=777, n_tasks=1)[0].gold
    task = generate_suite("reconciliation", seed=777, n_tasks=1)[0]
    results = EvalHarness(
        _ScriptedRunner({**gold, "confidence": 0.01}), max_steps=1
    ).run_suite([task], trials=1, seed=0)
    assert results[0].success
