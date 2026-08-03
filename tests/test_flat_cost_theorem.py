"""H0: the degenerate-case theorem test.

Claim (design spec section 2.1, C2): when all severities have equal cost
(flat K), the loss ranking of models MUST equal the accuracy ranking
(error-rate ranking). Every evaluation suite in the wild measures exactly
this flat-K special case while calling it a general accuracy result.

This test makes the claim executable: exhaustive over synthetic models and
seeds, using the shipped `flat` cost profile.
"""

from __future__ import annotations

import random

import pytest

from lossbench.costs.registry import load_cost_profile
from lossbench.metrics.loss import severity_weighted_loss
from lossbench.schema import Severity


def _synthetic_models(seed: int, n_models: int, n_tasks: int) -> list[list[bool]]:
    """Deterministic per-model error patterns with distinct error rates."""
    rng = random.Random(seed)
    error_rates = sorted(rng.random() for _ in range(n_models))
    return [
        [rng.random() < rate for _ in range(n_tasks)] for rate in error_rates
    ]


def _error_rate(errors: list[bool]) -> float:
    return sum(errors) / len(errors)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n_models", [2, 3])
def test_flat_cost_ranking_equals_accuracy_ranking(seed: int, n_models: int):
    flat = load_cost_profile("flat")
    n_tasks = 200
    # balanced severity mix so ranking isn't an artifact of one class
    severities = [
        Severity.LOW,
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.CRITICAL,
    ] * (n_tasks // 4)

    model_errors = _synthetic_models(seed, n_models, n_tasks)

    def loss_for(errors: list[bool]) -> float:
        return severity_weighted_loss(errors, severities, flat)

    losses = [loss_for(errors) for errors in model_errors]
    rates = [_error_rate(errors) for errors in model_errors]

    # lower loss must correspond to lower error rate, exactly
    loss_order = sorted(range(n_models), key=lambda i: losses[i])
    rate_order = sorted(range(n_models), key=lambda i: rates[i])
    assert loss_order == rate_order, (
        f"flat-K loss ranking diverged from accuracy ranking: "
        f"loss={losses} rates={rates}"
    )
