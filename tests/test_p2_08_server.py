from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from lossbench.ledger.store import AuditLedger
from lossbench.schema import CostProfile, DecisionEvent, DecisionKind, PolicyBundle
from lossbench.server import TenantConfig, TenantStore, create_app

FLAT = CostProfile(
    id="test-flat",
    description="test profile",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 1.0},
    escalate_cost=1.0,
)


def permissive_bundle(policy_id, allowlist=None):
    return PolicyBundle(
        id=policy_id,
        cost_model_id="flat",
        escalation_threshold=1.0,
        allowlist=allowlist or [],
    )


def decide_body(tenant_id, tool, p=0.1):
    return {
        "tenant_id": tenant_id,
        "request": {
            "tenant_id": tenant_id,
            "task_type": "reconciliation",
            "trajectory_state": {"severity": "LOW"},
            "proposed_action": {"tool": tool},
            "risk_features": {"calibrated_p": p},
            "available_models": [],
            "policy_ref": "pol-1",
        },
    }


def make_event(event_id="ev-1", tenant_id="tenant-a"):
    return DecisionEvent(
        event_id=event_id,
        tenant_id=tenant_id,
        trace_id="tr-1",
        trajectory_id="traj-1",
        task_id="task-1",
        timestamp=datetime.now(UTC),
        input_snapshot_hash="abc",
        prompt_hash="abc",
        model_id="m1",
        decision=DecisionKind.ALLOW,
        policy_id="pol-a",
        cost_model_id="flat",
    )


def test_health():
    store = TenantStore()
    with TestClient(create_app(store)) as client:
        resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "tenants": 0}


def test_decide_allowed():
    store = TenantStore()
    store.register(TenantConfig("tenant-a", permissive_bundle("pol-a"), FLAT))
    with TestClient(create_app(store)) as client:
        resp = client.post("/v1/decide", json=decide_body("tenant-a", "read"))
    assert resp.status_code == 200
    assert resp.json()["decision"] == DecisionKind.ALLOW.value


def test_decide_tenant_isolation():
    store = TenantStore()
    store.register(
        TenantConfig("tenant-a", permissive_bundle("pol-a", allowlist=["tool_x"]), FLAT)
    )
    deny_b = PolicyBundle(
        id="pol-b",
        cost_model_id="flat",
        escalation_threshold=1.0,
        deny=["tool_x"],
    )
    store.register(TenantConfig("tenant-b", deny_b, FLAT))
    with TestClient(create_app(store)) as client:
        allowed = client.post("/v1/decide", json=decide_body("tenant-a", "tool_x"))
        denied = client.post("/v1/decide", json=decide_body("tenant-b", "tool_x"))
    assert allowed.status_code == 200
    assert denied.status_code == 200
    assert allowed.json()["decision"] == DecisionKind.ALLOW.value
    assert denied.json()["decision"] == DecisionKind.DENY.value


def test_decide_unknown_tenant_403():
    store = TenantStore()
    with TestClient(create_app(store)) as client:
        resp = client.post("/v1/decide", json=decide_body("ghost", "read"))
    assert resp.status_code == 403


def test_decide_invalid_request_400():
    store = TenantStore()
    store.register(TenantConfig("tenant-a", permissive_bundle("pol-a"), FLAT))
    body = decide_body("tenant-a", "read")
    del body["request"]["policy_ref"]
    with TestClient(create_app(store)) as client:
        resp = client.post("/v1/decide", json=body)
    assert resp.status_code == 400


def test_config_endpoint():
    store = TenantStore()
    bundle = permissive_bundle("pol-a", allowlist=["read", "post"])
    store.register(TenantConfig("tenant-a", bundle, FLAT))
    with TestClient(create_app(store)) as client:
        resp = client.get("/v1/tenants/tenant-a/config")
    assert resp.status_code == 200
    assert resp.json() == {
        "policy_id": "pol-a",
        "cost_model_id": "flat",
        "escalation_threshold": 1.0,
        "allowlist": ["read", "post"],
    }


def test_events_endpoint_501_without_ledger():
    store = TenantStore()
    store.register(TenantConfig("tenant-a", permissive_bundle("pol-a"), FLAT))
    event = make_event()
    with TestClient(create_app(store)) as client:
        resp = client.post(
            "/v1/tenants/tenant-a/events", json=event.model_dump(mode="json")
        )
    assert resp.status_code == 501


def test_events_endpoint_201_with_ledger():
    store = TenantStore()
    store.register(TenantConfig("tenant-a", permissive_bundle("pol-a"), FLAT))
    ledger = AuditLedger()
    event = make_event(event_id="ev-1")
    with TestClient(create_app(store, ledger)) as client:
        resp = client.post(
            "/v1/tenants/tenant-a/events", json=event.model_dump(mode="json")
        )
    assert resp.status_code == 201
    stored = ledger.get("ev-1")
    assert stored is not None
    assert stored.event_id == "ev-1"
    assert stored.tenant_id == "tenant-a"


def test_duplicate_tenant_raises():
    store = TenantStore()
    store.register(TenantConfig("tenant-a", permissive_bundle("pol-a"), FLAT))
    with pytest.raises(ValueError):
        store.register(TenantConfig("tenant-a", permissive_bundle("pol-b"), FLAT))


def test_health_counts_tenants():
    store = TenantStore()
    store.register(TenantConfig("tenant-a", permissive_bundle("pol-a"), FLAT))
    store.register(TenantConfig("tenant-b", permissive_bundle("pol-b"), FLAT))
    with TestClient(create_app(store)) as client:
        resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["tenants"] == 2
