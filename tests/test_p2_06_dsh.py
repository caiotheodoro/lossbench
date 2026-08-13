"""Tests for the P2.6 dsh plugin bridge, manifest, and payloads."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lossbench.adapters.dsh import (
    DshPluginBridge,
    ToolDeniedError,
    build_manifest,
    hook_payload,
    manifest_json,
)
from lossbench.policy import PolicyEngine
from lossbench.record import TrajectoryRecorder
from lossbench.schema import CostProfile, DecisionEvent, DecisionKind, PolicyBundle

FLAT = CostProfile(
    id="flat",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 1.0},
)

MESSAGES = [{"role": "user", "content": "reconcile ledger 42"}]


def make_engine(*, escalation_threshold=1.0, allowlist=(), deny=(), model_tiers=()):
    bundle = PolicyBundle(
        id="p2.6",
        cost_model_id="flat",
        escalation_threshold=escalation_threshold,
        allowlist=list(allowlist),
        deny=list(deny),
        model_tiers=dict(model_tiers),
    )
    return PolicyEngine(bundle, FLAT)


def _event(decision: DecisionKind = DecisionKind.ALLOW, event_id: str = "evt-1") -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        trace_id="trace-1",
        trajectory_id="traj-1",
        task_id="task-1",
        timestamp=datetime.now(UTC),
        input_snapshot_hash="in-1",
        prompt_hash="pr-1",
        model_id="m1",
        decision=decision,
        policy_id="p1",
        cost_model_id="flat",
    )


def test_manifest_shape():
    manifest = build_manifest(
        "lossbench-risk-control",
        "0.1.0",
        ["beforeModel", "beforeTool", "onResolution"],
        "http://127.0.0.1:8090/lossbench",
    )
    assert set(manifest) == {"id", "version", "hooks", "bridge"}
    assert set(manifest["bridge"]) == {"url", "auth_header", "timeout_s"}
    assert manifest["id"] == "lossbench-risk-control"
    assert manifest["version"] == "0.1.0"
    assert manifest["hooks"] == ["beforeModel", "beforeTool", "onResolution"]
    assert manifest["bridge"]["url"] == "http://127.0.0.1:8090/lossbench"
    assert manifest["bridge"]["auth_header"] == "x-lossbench-key"
    assert manifest["bridge"]["timeout_s"] == 30


def test_manifest_json_deterministic_and_sorted():
    manifest = build_manifest(
        "lossbench-risk-control", "0.1.0", ["beforeModel"], "http://x/lossbench"
    )
    first = manifest_json(manifest)
    assert first == manifest_json(manifest)
    assert json.loads(first) == manifest
    assert first == json.dumps(manifest, sort_keys=True, indent=2)


def test_hook_payload_keys():
    payload = hook_payload("beforeModel")
    assert set(payload) == {"hook", "event", "tool_name", "args", "messages"}
    assert payload["event"] is None
    assert payload["tool_name"] is None
    assert payload["args"] is None
    assert payload["messages"] is None
    populated = hook_payload(
        "beforeTool",
        event=_event(),
        tool_name="read",
        args={"calibrated_p": 0.1},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert set(populated) == {"hook", "event", "tool_name", "args", "messages"}
    assert populated["event"]["decision"] == "ALLOW"
    assert populated["event"]["event_id"] == "evt-1"
    assert populated["tool_name"] == "read"
    assert populated["args"] == {"calibrated_p": 0.1}
    assert populated["messages"] == [{"role": "user", "content": "hi"}]


def test_before_model_continue():
    recorder = TrajectoryRecorder()
    bridge = DshPluginBridge(make_engine(), recorder)
    envelope = bridge.on_before_model(MESSAGES)
    assert set(envelope) == {"action", "payload"}
    assert envelope["action"] == "continue"
    assert envelope["payload"]["decision"]["decision"] == "ALLOW"
    event = envelope["payload"]["event"]
    recorded = recorder.trajectory_events(event["trajectory_id"])
    assert len(recorded) == 1
    assert recorded[0].decision is DecisionKind.ALLOW
    assert recorded[0].policy_id == "p2.6"


def test_before_model_block():
    bridge = DshPluginBridge(make_engine(deny=["model_call"]))
    envelope = bridge.on_before_model(MESSAGES)
    assert set(envelope) == {"action", "reason"}
    assert envelope["action"] == "block"
    assert "model_call" in envelope["reason"]


def test_before_model_escalate():
    bridge = DshPluginBridge(make_engine(escalation_threshold=0.5))
    messages = [
        {
            "role": "user",
            "content": "reconcile ledger 42",
            "lossbench": {"calibrated_p": 0.9, "severity": "HIGH"},
        }
    ]
    envelope = bridge.on_before_model(messages)
    assert set(envelope) == {"action", "reason"}
    assert envelope["action"] == "escalate"
    assert "escalat" in envelope["reason"]


def test_before_tool_block_and_continue():
    recorder = TrajectoryRecorder()
    bridge = DshPluginBridge(make_engine(deny=["drop"]), recorder)
    blocked = bridge.on_before_tool("drop", {"calibrated_p": 0.2})
    assert set(blocked) == {"action", "reason"}
    assert blocked["action"] == "block"
    assert "drop" in blocked["reason"]
    error = ToolDeniedError("drop", blocked["reason"])
    assert error.tool_name == "drop"
    assert error.reason == blocked["reason"]
    allowed = bridge.on_before_tool("read", {"calibrated_p": 0.2})
    assert allowed["action"] == "continue"
    assert allowed["payload"]["event"]["decision"] == "ALLOW"
    allowlisted = DshPluginBridge(make_engine(allowlist=["post"]))
    assert allowlisted.on_before_tool("read", {})["action"] == "block"


def test_on_resolution_validates():
    recorder = TrajectoryRecorder()
    bridge = DshPluginBridge(make_engine(), recorder)
    approved = bridge.on_resolution(
        {"decision_id": "evt-9", "resolution": "APPROVE", "reviewer": "alice"}
    )
    assert approved["decision"] == "ALLOW"
    assert approved["event_id"] == "evt-9@resolved"
    rejected = bridge.on_resolution(
        {"decision_id": "evt-10", "resolution": "REJECT", "reviewer": "bob"}
    )
    assert rejected["decision"] == "DENY"
    with pytest.raises(ValueError, match="APPROVE, REJECT, AMEND"):
        bridge.on_resolution({"decision_id": "evt-11", "resolution": "MAYBE", "reviewer": "carol"})
    with pytest.raises(ValueError, match="amended_action"):
        bridge.on_resolution({"decision_id": "evt-12", "resolution": "AMEND", "reviewer": "carol"})
    amended = bridge.on_resolution(
        {
            "decision_id": "evt-13",
            "resolution": "AMEND",
            "reviewer": "carol",
            "amended_action": {"tool": "read"},
        }
    )
    assert amended["decision"] == "VERIFY"
    assert amended["proposed_action"] == {"tool": "read"}
    assert [event.decision.value for event in recorder.flush()] == ["ALLOW", "DENY", "VERIFY"]


def test_recorder_optional():
    bridge = DshPluginBridge(make_engine())
    envelope = bridge.on_before_model(MESSAGES)
    assert envelope["action"] == "continue"
    assert bridge.on_before_tool("read", {})["action"] == "continue"
    resolution = bridge.on_resolution(
        {"decision_id": "evt-20", "resolution": "APPROVE", "reviewer": "alice"}
    )
    assert resolution["decision"] == "ALLOW"
    assert resolution["event_id"] == "evt-20@resolved"


def test_manifest_file_exists_and_parses():
    path = Path(__file__).resolve().parents[1] / "packaging" / "dsh" / "plugin.manifest.json"
    assert path.is_file()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["id"] == "lossbench-risk-control"
    assert manifest["version"] == "0.1.0"
    assert manifest["hooks"] == ["beforeModel", "beforeTool", "onResolution"]
    assert manifest["bridge"]["url"] == "http://127.0.0.1:8090/lossbench"
    assert manifest["bridge"]["auth_header"] == "x-lossbench-key"
    assert manifest["bridge"]["timeout_s"] == 30
