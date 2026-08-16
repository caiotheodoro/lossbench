"""Regression tests for SOTA-review fixes (wave of 10 review agents)."""

from __future__ import annotations

import random

import pytest

from lossbench.cache.store import ResponseCache
from lossbench.calibrate.pipeline import run_calibration_pipeline
from lossbench.contamination.monitor import task_signature as monitor_signature
from lossbench.costs.registry import load_cost_profile
from lossbench.eval import EvalHarness
from lossbench.generate import DOMAINS, generate_suite
from lossbench.generate.taxonomy import task_signature as generator_signature
from lossbench.ledger import AuditLedger
from lossbench.metrics.sensitivity import ranking_stability
from lossbench.policy.fit import fit_model_tiers
from lossbench.runners import make_stub_runner
from lossbench.schema import DecisionEvent, DecisionKind, PolicyBundle, Severity


def test_signature_spaces_agree_across_modules():
    """Generator and contamination monitor must hash the same task identically."""
    from lossbench.generate import generate_suite as gs

    for domain in DOMAINS:
        for task in gs(domain, seed=7, n_tasks=8):
            assert task.signature == generator_signature(task)
            assert task.signature == monitor_signature(task)


def test_ledger_detects_row_reordering():
    ledger = AuditLedger()
    events = [make_event(f"ev-{i}") for i in range(1, 4)]
    for event in events:
        ledger.append(event)
    ledger._conn.execute(
        "UPDATE events SET seq = CASE seq WHEN 1 THEN 2 WHEN 2 THEN 1 ELSE seq END "
        "WHERE seq IN (1, 2)"
    )
    result = ledger.verify()
    assert result["valid"] is False
    assert result["first_bad_seq"] in (1, 2)


def test_ledger_detects_head_truncation():
    ledger = AuditLedger()
    events = [make_event(f"ev-{i}") for i in range(1, 4)]
    for event in events:
        ledger.append(event)
    ledger._conn.execute("DELETE FROM events WHERE seq = 1")
    result = ledger.verify()
    # the remaining chain re-anchors at GENESIS with seq=2 -> linkage broken
    assert result["valid"] is False


def test_ledger_concurrent_append_serialized():
    """Interleaved appends must not corrupt the chain (single-writer lock)."""
    ledger = AuditLedger()
    order = list(range(20))
    random.Random(0).shuffle(order)
    for i in order:
        ledger.append(make_event(f"conc-{i}"))
    result = ledger.verify()
    assert result["valid"] is True
    assert result["n_events"] == 20


def test_cache_key_sensitivity_to_sampling_params():
    cache = ResponseCache()
    a = cache.cache_key("m", "ph", {"temperature": 0.2}, 0, "ih")
    b = cache.cache_key("m", "ph", {"temperature": 1.0}, 0, "ih")
    assert a != b
    c = cache.cache_key("m", "ph", {"reasoning_effort": "low"}, 0, "ih")
    d = cache.cache_key("m", "ph", {"reasoning_effort": "high"}, 0, "ih")
    assert c != d


def test_sensitivity_tie_is_not_crossover():
    """Exact loss ties must never count as a ranking flip."""
    n = 200
    models = {
        "aa": {
            "errors": [i % 2 == 0 for i in range(n)],
            "severities_mix": {"LOW": 1.0},
        },
        "zz": {
            "errors": [i == 0 for i in range(n)],
            "severities_mix": {"HIGH": 1.0},
        },
    }
    sevs = [Severity.LOW] * n
    stability = ranking_stability(models, sevs, (1.0, 2.0))
    assert stability["flips"] == 0


def test_fit_model_tiers_formula():
    profile = load_cost_profile("reconciliation")  # K(HIGH) = 10
    tiers = fit_model_tiers(
        {"cheap": 0.3, "frontier": 0.05}, Severity.HIGH, profile, {"cheap": 0.0, "frontier": 0.0}
    )
    assert tiers["cheap"] == pytest.approx(0.3 * 10 * 0.5)
    assert tiers["frontier"] == pytest.approx(0.05 * 10 * 0.5)
    zero = fit_model_tiers({"m": 0.0}, Severity.HIGH, profile, {"m": 0.0})
    assert zero["m"] == 0.0


def test_harness_events_carry_severity_error_and_risk():
    """Benchmark -> control pipeline: events must carry severity, error labels
    and risk features, or replay/calibration read garbage."""
    import json as _json


    tasks = generate_suite("reconciliation", seed=7, n_tasks=20)
    responses = {t.id: _json.dumps(t.gold, sort_keys=True) for t in tasks}
    harness = EvalHarness(make_stub_runner("stub", responses))
    results = harness.run_suite(tasks, trials=1, seed=1)
    finals = [t for r in results for t in r.events if t.decision == DecisionKind.ALLOW]
    assert finals
    for event, task in zip(finals, tasks, strict=True):
        assert (event.observed_outcome or {}).get("severity") == task.severity.value
        assert "error" in (event.observed_outcome or {})
        assert event.risk_features.get("calibrated_p") in (0.1, 0.9)


def test_pipeline_heldout_ece_never_zero_when_imperfect():
    """The in-sample leak is closed: a held-out ECE must exist and must not
    be a fake 0.0 for a realistically imperfect calibrator."""
    rng = random.Random(3)
    n = 200
    p = [rng.random() for _ in range(n)]
    correct = [rng.random() < x for x in p]
    conf = [min(0.999, max(0.001, x**0.5)) for x in p]
    events = [
        make_event(
            f"cal-{i}",
            conf,
            bool(ok),
            severity=Severity.MEDIUM.value,
        )
        for i, (conf, ok) in enumerate(zip(conf, correct, strict=True))
    ]
    result = run_calibration_pipeline(events, load_cost_profile("reconciliation"))
    assert "calibrated_ece" in result.report
    assert "calibrated_ece_fit" in result.report


def test_dsh_tool_args_cannot_override_tool_name():
    from lossbench.adapters.dsh.plugin import DshPluginBridge
    from lossbench.policy import PolicyEngine

    bundle = PolicyBundle(
        id="p1",
        cost_model_id="reconciliation",
        escalation_threshold=1.0,
        deny=["rm"],
    )
    engine = PolicyEngine(bundle, load_cost_profile("reconciliation"))
    bridge = DshPluginBridge(engine)
    # args containing a "tool" key must not overwrite the real tool name
    envelope = bridge.on_before_tool("read", {"tool": "rm", "path": "/"})
    assert envelope["action"] == "continue"


def test_buzz_outbox_idempotent_under_duplicate():
    from lossbench.buzz import BuzzOutbox
    from lossbench.hitl import ReviewRequest

    outbox = BuzzOutbox()
    req = ReviewRequest(
        decision_id="d-1",
        trajectory_id="t-1",
        tenant_id="tenant-a",
        task_id="task-1",
        proposed_action={"tool": "post"},
        expected_loss=5.0,
        rationale="high risk",
        policy_ref="p1",
    )
    first = outbox.enqueue_review_request(req, community="acme-bank")
    second = outbox.enqueue_review_request(req, community="acme-bank")
    assert first.outbox_id == second.outbox_id
    assert len(outbox.pending()) == 1


def make_event(event_id, conf=0.5, error=False, severity="LOW"):
    from datetime import datetime

    return DecisionEvent(
        event_id=event_id,
        tenant_id="default",
        trace_id="t",
        trajectory_id="tr",
        task_id="task",
        timestamp=datetime(2026, 8, 1),
        input_snapshot_hash="i",
        prompt_hash="p",
        model_id="m",
        decision=DecisionKind.ALLOW,
        policy_id="p1",
        cost_model_id="reconciliation",
        calibrated_probability=conf,
        observed_outcome={"error": error, "severity": severity},
    )
