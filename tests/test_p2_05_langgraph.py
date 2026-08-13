"""Tests for the P2.5 LangGraph middleware adapter."""

from __future__ import annotations

import importlib
import sys

import pytest

from lossbench.adapters import LossGuardMiddleware, ToolDeniedError
from lossbench.policy import PolicyEngine
from lossbench.record import TrajectoryRecorder
from lossbench.schema import CostProfile, DecisionKind, PolicyBundle

FLAT = CostProfile(
    id="flat",
    description="d",
    severity_costs={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0, "CRITICAL": 1.0},
    escalate_cost=1.0,
)


def make_engine(*, deny=None, allowlist=None, threshold=0.7) -> PolicyEngine:
    bundle = PolicyBundle(
        id="pol-1",
        cost_model_id="flat",
        escalation_threshold=threshold,
        deny=list(deny or []),
        allowlist=list(allowlist or []),
    )
    return PolicyEngine(bundle, FLAT)


def test_deny_blocks_model_call():
    middleware = LossGuardMiddleware(make_engine(deny=["rm"]))
    params = {"model": "m1", "messages": ["hi"], "tools": ["rm"], "tool": "rm"}
    with pytest.raises(RuntimeError, match="denied"):
        middleware.before_model(params)


def test_escalate_records_not_blocks():
    recorder = TrajectoryRecorder()
    middleware = LossGuardMiddleware(make_engine(threshold=0.3), recorder=recorder)
    params = {
        "model": "m1",
        "messages": ["risky step"],
        "tools": ["post"],
        "tool": "post",
        "calibrated_p": 0.9,
    }
    assert middleware.before_model(params) is params
    events = recorder.flush()
    assert len(events) == 1
    assert events[0].decision == DecisionKind.ESCALATE


def test_allow_passes_through():
    recorder = TrajectoryRecorder()
    middleware = LossGuardMiddleware(make_engine(), recorder=recorder)
    params = {"model": "m1", "messages": ["hello"], "tools": []}
    assert middleware.before_model(params) is params
    events = recorder.flush()
    assert len(events) == 1
    assert events[0].decision == DecisionKind.ALLOW


class FakeResponse:
    model = "fake-model"
    usage = {"prompt_tokens": 10}


def test_after_model_does_not_double_record():
    # Recording discipline: exactly ONE event per decision point, recorded by
    # the gate hook. after_model is a no-op and must not fabricate a second
    # ALLOW event (which would contradict an ESCALATE).
    recorder = TrajectoryRecorder()
    middleware = LossGuardMiddleware(make_engine(), recorder=recorder)
    params = {"model": "m1", "messages": ["hello"], "tools": []}
    middleware.before_model(params)
    response = FakeResponse()
    assert middleware.after_model(response) is response
    events = recorder.flush()
    assert len(events) == 1
    assert events[0].decision == DecisionKind.ALLOW
    assert events[0].model_id == "m1"


def test_before_tool_denied():
    middleware = LossGuardMiddleware(make_engine(deny=["rm"]))
    with pytest.raises(ToolDeniedError) as exc:
        middleware.before_tool("rm", {"path": "/"})
    assert exc.value.tool_name == "rm"


def test_before_tool_allowed():
    middleware = LossGuardMiddleware(make_engine(deny=["rm"]))
    args = {"path": "/tmp"}
    assert middleware.before_tool("read", args) is args


def test_to_request_shape():
    middleware = LossGuardMiddleware(make_engine(), task_type="reconciliation")
    params = {"model": "m1", "messages": ["hello", "world"], "tools": ["read", "post"]}
    request = middleware.to_request(params)
    assert request.proposed_action["model"] == "m1"
    assert request.proposed_action["tools"] == ["read", "post"]
    assert request.proposed_action["tool"] == "model_call"
    # messages are never persisted raw: only their SHA-256 travels
    h = request.trajectory_state["last_message_hash"]
    assert len(h) == 64
    assert "world" not in request.trajectory_state
    assert request.task_type == "reconciliation"
    assert request.tenant_id == "default"


def test_no_langgraph_import_at_module_load():
    module = importlib.import_module("lossbench.adapters.langgraph")
    importlib.reload(module)
    assert "langgraph" not in sys.modules


def test_recorder_optional():
    middleware = LossGuardMiddleware(make_engine())
    params = {"model": "m1", "messages": ["hi"], "tools": []}
    assert middleware.before_model(params) is params
    response = FakeResponse()
    assert middleware.after_model(response) is response
    assert middleware.before_tool("read", {}) == {}
