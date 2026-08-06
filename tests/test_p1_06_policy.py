import pytest

from lossbench.policy import (
    PolicyEngine,
    dump_policy,
    fit_escalation_threshold,
    load_policy,
)
from lossbench.schema import CostProfile, DecisionKind, DecisionRequest, PolicyBundle, Severity

FLAT = CostProfile(
    id="test-flat",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 1.0},
    escalate_cost=1.0,
)

ASYMMETRIC = CostProfile(
    id="test-asym",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 10.0, "CRITICAL": 50.0},
    escalate_cost=1.0,
)

REVIEW_HEAVY = CostProfile(
    id="test-review-heavy",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 5.0, "HIGH": 50.0, "CRITICAL": 250.0},
    escalate_cost=25.0,
)


def make_request(tool="read", p=0.1, severity="LOW", models=None):
    return DecisionRequest(
        tenant_id="t1",
        task_type="reconciliation",
        trajectory_state={"severity": severity},
        proposed_action={"tool": tool},
        risk_features={"calibrated_p": p},
        available_models=list(models or []),
        policy_ref="policy-1",
    )


def test_policy_roundtrip_yaml(tmp_path):
    bundle = PolicyBundle(
        id="rt-1",
        revision="2.0.0",
        cost_model_id="flat",
        escalation_threshold=0.7,
        route_thresholds={"HIGH": 0.3},
        model_tiers={"cheap": 0.1, "frontier": 1.0},
        spend_cap=50.0,
        latency_sla_s=10.0,
        allowlist=["read", "post"],
        deny=["drop"],
    )
    path = tmp_path / "policy.yaml"
    dump_policy(bundle, path)
    assert load_policy(path) == bundle


def test_deny_rule_first():
    bundle = PolicyBundle(
        id="p1", cost_model_id="flat", escalation_threshold=0.7, deny=["drop"]
    )
    engine = PolicyEngine(bundle, FLAT)
    resp = engine.decide(make_request(tool="drop", p=0.95))
    assert resp.decision == DecisionKind.DENY
    assert resp.requires_human is False


def test_allowlist_deny_when_not_listed():
    bundle = PolicyBundle(
        id="p1", cost_model_id="flat", escalation_threshold=0.7, allowlist=["post"]
    )
    engine = PolicyEngine(bundle, FLAT)
    resp = engine.decide(make_request(tool="read"))
    assert resp.decision == DecisionKind.DENY


def test_escalation_by_threshold():
    bundle = PolicyBundle(id="p1", cost_model_id="flat", escalation_threshold=0.7)
    engine = PolicyEngine(bundle, FLAT)
    resp = engine.decide(make_request(tool="post", p=0.9))
    assert resp.decision == DecisionKind.ESCALATE
    assert resp.requires_human is True
    assert resp.expected_loss == pytest.approx(0.9)


def test_route_uses_bayes():
    # Review-heavy environment (escalate_cost 25): HIGH-severity expected
    # loss (p*K <= 9) stays below review cost, so the decision ROUTES between
    # model tiers instead of escalating.
    bundle = PolicyBundle(
        id="p1",
        cost_model_id="flat",
        escalation_threshold=1.0,
        model_tiers={"cheap": 0.1, "frontier": 1.0},
    )
    engine = PolicyEngine(bundle, REVIEW_HEAVY)
    hi = engine.decide(
        make_request(tool="post", p=0.9, severity="MEDIUM", models=["cheap", "frontier"])
    )
    assert hi.decision == DecisionKind.ROUTE
    assert hi.selected_model == "frontier"
    lo = engine.decide(
        make_request(tool="post", p=0.05, severity="MEDIUM", models=["cheap", "frontier"])
    )
    assert lo.decision == DecisionKind.ROUTE
    assert lo.selected_model == "cheap"


def test_bayes_guard_escalates_low_p_high_severity():
    # The flagship fix: a CRITICAL case at p=0.3 has expected loss 15.0,
    # well above the 1.0 review cost, so it ESCALATES even below any p
    # threshold. Threshold-only policies would auto-approve it.
    bundle = PolicyBundle(
        id="p1",
        cost_model_id="flat",
        escalation_threshold=0.9,
        model_tiers={"cheap": 0.1, "frontier": 1.0},
    )
    engine = PolicyEngine(bundle, ASYMMETRIC)
    resp = engine.decide(
        make_request(tool="post", p=0.3, severity="CRITICAL", models=["cheap", "frontier"])
    )
    assert resp.decision == DecisionKind.ESCALATE
    assert resp.requires_human is True
    assert resp.expected_loss == pytest.approx(15.0)


def test_allow_fallback():
    bundle = PolicyBundle(id="p1", cost_model_id="flat", escalation_threshold=1.0)
    engine = PolicyEngine(bundle, FLAT)
    resp = engine.decide(make_request(tool="read", p=0.2))
    assert resp.decision == DecisionKind.ALLOW
    assert resp.requires_human is False


def test_fit_escalation_threshold_lower_for_expensive_errors():
    probs = [0.30, 0.35, 0.40, 0.70, 0.85]
    errors = [True, True, True, False, False]
    sevs = [Severity.HIGH, Severity.HIGH, Severity.HIGH, Severity.LOW, Severity.LOW]
    flat = fit_escalation_threshold(probs, errors, sevs, FLAT)
    high_k = fit_escalation_threshold(probs, errors, sevs, ASYMMETRIC)
    assert high_k["best_threshold"] <= flat["best_threshold"]
    assert flat["best_threshold"] == pytest.approx(0.875)


def test_fit_returns_documented_keys():
    probs = [0.1, 0.5, 0.9]
    errors = [False, True, True]
    sevs = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]
    res = fit_escalation_threshold(probs, errors, sevs, ASYMMETRIC)
    assert set(res) == {"best_threshold", "best_cost", "baseline_cost", "n"}
    assert res["n"] == 3
    assert res["baseline_cost"] == pytest.approx(11.0)


def test_load_policy_invalid_file_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: [unclosed\n  : 1\n")
    with pytest.raises(ValueError):
        load_policy(bad)


def test_load_policy_unknown_field_raises(tmp_path):
    bad = tmp_path / "unknown.yaml"
    bad.write_text(
        "id: p1\ncost_model_id: flat\nescalation_threshold: 0.5\nbogus_field: 1\n"
    )
    with pytest.raises(ValueError):
        load_policy(bad)


def test_engine_deterministic():
    bundle = PolicyBundle(
        id="p1",
        cost_model_id="flat",
        escalation_threshold=0.7,
        model_tiers={"cheap": 0.1, "frontier": 1.0},
    )
    engine = PolicyEngine(bundle, ASYMMETRIC)
    request = make_request(tool="post", p=0.8, severity="HIGH", models=["cheap", "frontier"])
    assert engine.decide(request) == engine.decide(request)
