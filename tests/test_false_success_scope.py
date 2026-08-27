"""Regression: false_success_rate must fire for claim-then-verify sources.

Issue #10 — the metric was a provable constant 0.0 because the only
fixtures exercising it were harness-shaped (every event pre-verified
against gold, so observed_outcome was never None). The dsh adapter records
an ALLOW *independently* of any verification, so a genuine
unverified-success trajectory can occur there. This test feeds one in and
proves the detector returns nonzero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lossbench.adapters.dsh import DshPluginBridge
from lossbench.policy import PolicyEngine
from lossbench.record import TrajectoryRecorder
from lossbench.schema import CostProfile, DecisionEvent, DecisionKind, PolicyBundle
from lossbench.scoring.false_success import (
    TrajectorySource,
    false_success_applicable,
    false_success_rate,
)

FLAT = CostProfile(
    id="flat",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 1.0},
)
MESSAGES = [{"role": "user", "content": "reconcile ledger 42"}]


def _engine() -> PolicyEngine:
    bundle = PolicyBundle(
        id="issue-10",
        cost_model_id="flat",
        escalation_threshold=1.0,
        allowlist=[],
        deny=[],
        model_tiers={},
    )
    return PolicyEngine(bundle, FLAT)


def _state_unchanged(events: Sequence[DecisionEvent], gold: dict[str, Any]) -> bool:
    return False


def _state_reproduced(events: Sequence[DecisionEvent], gold: dict[str, Any]) -> bool:
    return True


def test_capability_marker():
    assert false_success_applicable(TrajectorySource.CLAIM_THEN_VERIFY) is True
    assert false_success_applicable(TrajectorySource.SELF_VERIFYING) is False


def test_dsh_unverified_claim_scores_nonzero():
    recorder = TrajectoryRecorder()
    bridge = DshPluginBridge(_engine(), recorder)

    envelope = bridge.on_before_model(MESSAGES)
    assert envelope["action"] == "continue"

    events = recorder.flush()
    assert len(events) == 1
    claim = events[0]
    # dsh records the ALLOW claim with no outcome attached — nothing verified it.
    assert claim.decision is DecisionKind.ALLOW
    assert claim.observed_outcome is None

    trajectories = [events]
    golds = [{"ledger": "reconciled"}]

    assert false_success_rate(trajectories, golds, _state_unchanged) == 1.0
    # And it is a genuine detector, not a constant: a claim that reproduces
    # the gold state is not a false success.
    assert false_success_rate(trajectories, golds, _state_reproduced) == 0.0
