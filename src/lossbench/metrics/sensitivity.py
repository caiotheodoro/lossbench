"""Cost-sensitivity analysis: how model rankings shift as severity costs change.

The analysis is about the METRIC's behavior under cost regimes, not model
realism: models are explicit synthetic error patterns, so the result is
"under which cost ratios does the loss ranking swap", which is exactly the
contestable-input property the benchmark promises. Scaling rule: for ratio r,
severity_costs[HIGH] *= r and severity_costs[CRITICAL] *= r; LOW/MEDIUM are
unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lossbench.costs.registry import load_cost_profile
from lossbench.metrics.loss import severity_weighted_loss
from lossbench.schema import CostProfile, Severity

_HIGH = Severity.HIGH.value
_CRITICAL = Severity.CRITICAL.value


def _severities_for(model: dict[str, Any], n: int) -> list[Severity]:
    """Derive per-task severities from a model's severity mix, deterministically."""
    mix = model.get("severities_mix") or {sev.value: 1.0 for sev in Severity}
    total = sum(mix.values())
    out: list[Severity] = []
    for sev in Severity:
        count = round(n * mix.get(sev.value, 0.0) / total)
        out.extend([sev] * count)
    return out[:n]


def _scaled_profile(base: CostProfile, ratio: float) -> CostProfile:
    profile = base.model_copy(deep=True)
    profile.severity_costs[_HIGH] = base.severity_costs[_HIGH] * ratio
    profile.severity_costs[_CRITICAL] = base.severity_costs[_CRITICAL] * ratio
    return profile


def cost_sensitivity_curves(
    models: dict[str, dict[str, Any]],
    severities: Sequence[Severity],
    cost_ratios: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 100.0),
    base_profile: CostProfile | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Per model, per ratio: severity-weighted loss over its error pattern."""
    base = base_profile or load_cost_profile("reconciliation")
    n = len(severities)
    curves: dict[str, list[dict[str, float]]] = {}
    for model_id, model in models.items():
        errors = model["errors"]
        if len(errors) != n:
            raise ValueError(
                f"model '{model_id}' error pattern length {len(errors)} != {n}"
            )
        per_ratio: list[dict[str, float]] = []
        for ratio in cost_ratios:
            profile = _scaled_profile(base, ratio)
            sevs = _severities_for(model, n)
            loss = severity_weighted_loss(errors, sevs, profile)
            per_ratio.append({"ratio": float(ratio), "loss": round(loss, 4)})
        curves[model_id] = per_ratio
    return curves


def ranking_stability(
    models: dict[str, dict[str, Any]],
    severities: Sequence[Severity],
    cost_ratios: Sequence[float],
    base_profile: CostProfile | None = None,
) -> dict:
    """Crossovers, flips, and per-ratio loss rankings.

    crossover_ratios maps "(modelA,modelB)" (lexicographically ordered) to the
    first ratio at which their loss ranking swaps, or None if never.
    """
    curves = cost_sensitivity_curves(models, severities, cost_ratios, base_profile)
    model_ids = sorted(curves)
    rankings: dict[float, list[str]] = {}
    for point_idx, ratio in enumerate(cost_ratios):
        ranked = sorted(
            model_ids,
            key=lambda m: (curves[m][point_idx]["loss"], m),
        )
        rankings[float(ratio)] = ranked

    crossovers: dict[str, float | None] = {}
    for i, a in enumerate(model_ids):
        for b in model_ids[i + 1 :]:
            key = f"({a},{b})" if a < b else f"({b},{a})"
            # Strict comparison on unrounded losses: an exact tie is NO
            # crossover (alphabetical tie-breaks would report a phantom flip).
            loss_a1 = curves[a][0]["loss"]
            loss_b1 = curves[b][0]["loss"]
            if loss_a1 == loss_b1:
                crossovers[key] = None
                continue
            a_wins_at_1 = loss_a1 < loss_b1
            crossover = None
            for point_idx, ratio in enumerate(cost_ratios):
                loss_a = curves[a][point_idx]["loss"]
                loss_b = curves[b][point_idx]["loss"]
                if loss_a == loss_b:
                    continue
                a_wins_now = loss_a < loss_b
                if a_wins_now != a_wins_at_1:
                    crossover = float(ratio)
                    break
            crossovers[key] = crossover

    return {
        "crossover_ratios": crossovers,
        "flips": sum(1 for v in crossovers.values() if v is not None),
        "rankings": rankings,
    }


def model_wins_share(
    models: dict[str, dict[str, Any]],
    severities: Sequence[Severity],
    cost_ratios: Sequence[float],
    base_profile: CostProfile | None = None,
) -> dict[str, float]:
    """Share of cost ratios at which each model has the lowest loss."""
    stability = ranking_stability(models, severities, cost_ratios, base_profile)
    wins = {model_id: 0.0 for model_id in stability["rankings"].get(cost_ratios[0], [])}
    for ranked in stability["rankings"].values():
        wins[ranked[0]] += 1.0 / len(stability["rankings"])
    return wins
