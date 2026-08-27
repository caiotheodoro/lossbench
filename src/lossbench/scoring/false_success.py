"""False-success detection over DecisionEvent trajectories.

Pure functions; no I/O. A trajectory that ends in an ALLOW/VERIFY claim
without a recorded outcome is only credited if its recorded actions
reproduce the gold final state; otherwise it is a false success.

Scope — which trajectory sources this metric applies to
------------------------------------------------------
The detector fires only when a trajectory's final event is an ALLOW/VERIFY
claim AND ``observed_outcome is None`` — i.e. the agent asserted completion
and nothing recorded an outcome for that claim. That precondition is
structural, not statistical: it can only ever occur for a
``CLAIM_THEN_VERIFY`` trajectory source (the dsh and langgraph adapters),
where an ALLOW is recorded before — and independently of — any
verification.

For a ``SELF_VERIFYING`` source (``lossbench.eval.harness.EvalHarness``)
every event is verified against gold by construction and that result is
folded into ``observed_outcome`` on every event, so ``observed_outcome`` is
never ``None`` and ``false_success_rate`` is a provable constant 0.0. A
0.0 from such a run is an artifact of the data source, not a measurement,
so harness-scored runs must report this metric as not-applicable rather
than as a clean 0.0 (see ``false_success_applicable`` and
``summarize_suite``). Feeding harness trajectories here still returns a
correct 0.0; the point is not to headline that number.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

from lossbench.schema import DecisionEvent, DecisionKind

CLAIM_DECISIONS = (DecisionKind.ALLOW, DecisionKind.VERIFY)


class TrajectorySource(StrEnum):
    """Capability marker for how a trajectory source records outcomes.

    ``SELF_VERIFYING`` — every event carries a gold-verified
    ``observed_outcome`` by construction (``EvalHarness``); an unverified
    success claim can never occur, so ``false_success_rate`` is a constant
    0.0 and is not a meaningful measurement.

    ``CLAIM_THEN_VERIFY`` — an ALLOW/VERIFY claim is recorded before, and
    independently of, any verification (the dsh and langgraph adapters), so
    a genuine unverified-claim trajectory can occur and
    ``false_success_rate`` is a real detector.
    """

    SELF_VERIFYING = "self_verifying"
    CLAIM_THEN_VERIFY = "claim_then_verify"


def false_success_applicable(source: TrajectorySource) -> bool:
    """Whether ``false_success_rate`` is a meaningful measurement for ``source``.

    False for ``SELF_VERIFYING`` sources, where the rate is a structural
    constant 0.0; True for ``CLAIM_THEN_VERIFY`` sources.
    """
    return source is TrajectorySource.CLAIM_THEN_VERIFY


def false_success_rate(
    trajectories: Sequence[Sequence[DecisionEvent]],
    gold_states: Sequence[dict[str, Any]],
    verify_state: Callable[[Sequence[DecisionEvent], dict[str, Any]], bool],
) -> float:
    """Share of trajectories that end in an unverified success claim.

    Only meaningful for ``CLAIM_THEN_VERIFY`` trajectory sources; see the
    module docstring and ``false_success_applicable``. For a
    ``SELF_VERIFYING`` source this is a provable constant 0.0.

    A trajectory claims success when its final event's decision is ALLOW
    or VERIFY and observed_outcome is absent (the agent asserted
    completion without recording an outcome). verify_state(events,
    gold_state) returns whether the trajectory's recorded actions
    reproduce the gold final state; a claim that fails verification is a
    false success — the 'agent claims done, state unchanged' detector.
    The rate is the share of ALL trajectories (not just claims):
    trajectories ending in other decisions (ESCALATE, DENY, ...) or with
    an observed outcome are safe by construction and count toward the
    denominator. Empty input yields 0.0; a length mismatch between
    trajectories and gold_states raises ValueError.
    """
    if len(trajectories) != len(gold_states):
        raise ValueError("trajectories and gold_states must have equal length")
    if not trajectories:
        return 0.0
    false = 0
    for events, gold in zip(trajectories, gold_states, strict=True):
        if not events:
            continue
        last = events[-1]
        if last.decision not in CLAIM_DECISIONS or last.observed_outcome is not None:
            continue
        if not verify_state(events, gold):
            false += 1
    return false / len(trajectories)
