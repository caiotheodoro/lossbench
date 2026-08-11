"""P1.13 cost-sensitivity analysis tests.

Crossover construction (n=6000, mix LOW 0.9 / HIGH 0.1):
  A: 100 LOW errors + 40 HIGH errors -> loss_A(r) = 20 + 400r
  B: 3000 LOW errors + 30 HIGH errors -> loss_B(r) = 600 + 300r
At r=1: A(420) < B(900) -> A wins. At r=10: A(4020) > B(3600) -> B wins.
Crossover between 5 and 10.
"""

from __future__ import annotations

import pytest

from lossbench.costs.registry import load_cost_profile
from lossbench.metrics.sensitivity import (
    cost_sensitivity_curves,
    model_wins_share,
    ranking_stability,
)
from lossbench.schema import Severity

N = 6000
RATIOS = (1.0, 2.0, 5.0, 10.0, 100.0)

SEVS = [Severity.LOW] * (N // 2) + [Severity.HIGH] * (N // 2)

# positions 0..5399 are LOW (mix 0.9), 5400..5999 are HIGH (mix 0.1)
MODELS_A_B = {
    "a_few_low_many_high": {
        "errors": [i < 100 or (5400 <= i < 5440) for i in range(N)],
        "severities_mix": {"LOW": 0.9, "HIGH": 0.1},
    },
    "b_many_low_few_high": {
        "errors": [i < 3000 or (5400 <= i < 5430) for i in range(N)],
        "severities_mix": {"LOW": 0.9, "HIGH": 0.1},
    },
}


def test_crossings_exist():
    stability = ranking_stability(MODELS_A_B, SEVS, RATIOS)
    assert stability["flips"] >= 1
    assert any(v is not None for v in stability["crossover_ratios"].values())
    wins = model_wins_share(MODELS_A_B, SEVS, RATIOS)
    assert wins["b_many_low_few_high"] > 0.0
    assert wins["a_few_low_many_high"] > 0.0


def test_b_wins_at_high_ratios_a_wins_at_low():
    curves = cost_sensitivity_curves(MODELS_A_B, SEVS, RATIOS)
    a = curves["a_few_low_many_high"]
    b = curves["b_many_low_few_high"]
    assert a[0]["loss"] < b[0]["loss"]
    assert a[3]["loss"] > b[3]["loss"]
    assert a[4]["loss"] > b[4]["loss"]


def test_single_model_no_crossover():
    single = {"only": MODELS_A_B["a_few_low_many_high"]}
    stability = ranking_stability(single, SEVS, RATIOS)
    assert stability["flips"] == 0
    assert all(v is None for v in stability["crossover_ratios"].values())


def test_loss_increases_with_ratio_for_high_errors():
    curves = cost_sensitivity_curves(MODELS_A_B, SEVS, RATIOS)
    losses = [p["loss"] for p in curves["a_few_low_many_high"]]
    assert losses == sorted(losses)
    assert losses[0] < losses[-1]


def test_flat_profile_no_crossings():
    flat = load_cost_profile("flat")
    stability = ranking_stability(MODELS_A_B, SEVS, RATIOS, base_profile=flat)
    assert stability["flips"] == 0


def test_output_shapes():
    curves = cost_sensitivity_curves(MODELS_A_B, SEVS, RATIOS)
    assert set(curves) == set(MODELS_A_B)
    assert len(curves["a_few_low_many_high"]) == len(RATIOS)
    assert set(curves["a_few_low_many_high"][0]) == {"ratio", "loss"}


def test_deterministic():
    assert cost_sensitivity_curves(MODELS_A_B, SEVS, RATIOS) == cost_sensitivity_curves(
        MODELS_A_B, SEVS, RATIOS
    )


def test_zero_error_model():
    models = {
        "perfect": {"errors": [False] * N, "severities_mix": {"HIGH": 1.0}}
    }
    curves = cost_sensitivity_curves(models, SEVS, RATIOS)
    assert all(p["loss"] == 0.0 for p in curves["perfect"])
    assert model_wins_share(models, SEVS, RATIOS)["perfect"] == 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="error pattern"):
        cost_sensitivity_curves({"bad": {"errors": [True] * 10}}, SEVS, RATIOS)
