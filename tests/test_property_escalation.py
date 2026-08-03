"""Property test: raising K(HIGH) never lowers the optimal escalation rate.

For a fixed calibrated risk population, the fraction of cases where
`escalate_iff` fires is monotone non-decreasing in the high-severity cost
(design spec section 2.1 and IMPLEMENTATION.md P0.6).
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from lossbench.decision import escalate_iff
from lossbench.schema import CostProfile, Severity


def _population(seed: int, n: int) -> list[tuple[float, Severity]]:
    rng = random.Random(seed)
    sevs = [Severity.HIGH] * 4 + [Severity.LOW, Severity.MEDIUM, Severity.CRITICAL]
    return [(rng.random(), sevs[rng.randrange(len(sevs))]) for _ in range(n)]


def _escalation_rate(profile: CostProfile, pop: list[tuple[float, Severity]]) -> float:
    fired = sum(1 for p, sev in pop if escalate_iff(p, sev, profile))
    return fired / len(pop)


@pytest.mark.parametrize("seed", [7, 11, 13])
def test_raising_high_cost_never_lowers_escalation_rate(seed: int):
    base = CostProfile(
        id="prop-test",
        description="d",
        severity_costs={"LOW": 0.2, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 10.0},
        escalate_cost=1.0,
    )
    pop = _population(seed, 500)

    rates = []
    for multiplier in (1.0, 2.0, 5.0, 10.0, 100.0):
        profile = base.model_copy(deep=True)
        profile.severity_costs[Severity.HIGH.value] = base.severity_costs[
            Severity.HIGH.value
        ] * multiplier
        rates.append(_escalation_rate(profile, pop))

    for prev, curr in pairwise(rates):
        assert curr >= prev - 1e-9, (
            f"escalation rate decreased as K(HIGH) rose: {rates}"
        )


def test_escalation_rate_increases_with_high_cost():
    """Same population must escalate at least as often at 100x high cost."""
    base = CostProfile(
        id="prop-test-2",
        description="d",
        severity_costs={"LOW": 0.2, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 10.0},
        escalate_cost=1.0,
    )
    pop = _population(42, 1000)
    low = _escalation_rate(base, pop)
    high = base.model_copy(deep=True)
    high.severity_costs[Severity.HIGH.value] = 100.0
    assert _escalation_rate(high, pop) >= low
