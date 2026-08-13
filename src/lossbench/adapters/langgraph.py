"""LangGraph/Deep Agents adapter: a policy gate over model and tool calls.

The adapter targets the real LangChain middleware protocol surface:
``AgentMiddleware``-compatible objects expose ``name``, ``wrap_model_call``
(request/response wrapper with ``request.override(model=...)`` for routing)
and ``wrap_tool_call`` (request wrapper whose denial is expressed by raising
``MiddlewareInterrupt`` or returning a synthetic result — never by crashing
the graph). LangChain is NOT imported at module load; the hooks are
duck-typed so the adapter is testable without the framework and compatible
with any harness exposing the same shape.

Recording discipline: exactly ONE DecisionEvent per decision point, recorded
in the model-call gate with the engine's actual decision. The after-hook
never fabricates a second ALLOW event (that would contradict an ESCALATE).
"""

from __future__ import annotations

import hashlib
from datetime import UTC
from typing import Any

from lossbench.policy.engine import PolicyEngine
from lossbench.record.recorder import TrajectoryRecorder
from lossbench.schema import DecisionKind, DecisionRequest, DecisionResponse

_MODEL_ACTION = "model_call"


class MiddlewareInterrupt(RuntimeError):
    """Sanctioned denial signal: the call must not proceed to the model/tool.

    Framework adapters translate this into their native block primitive
    (LangGraph's ``interrupt``/``Command``, a synthetic ``ToolMessage``, or a
    short-circuit); the adapter itself never lets it crash the graph.
    """


class ToolDeniedError(MiddlewareInterrupt):
    """Denial of a specific tool call by policy."""

    def __init__(self, tool_name: str, reason: str = ""):
        super().__init__(f"tool '{tool_name}' denied by policy: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class LossGuardMiddleware:
    """Policy gate for agent harnesses exposing LangChain-style middleware.

    Implements the ``AgentMiddleware`` surface: ``name`` plus
    ``wrap_model_call`` and ``wrap_tool_call``. ``before_model`` /
    ``before_tool`` / ``after_model`` remain as thin aliases for tests and
    for harnesses with the older hook vocabulary.
    """

    name = "lossbench-risk-control"

    def __init__(
        self,
        engine: PolicyEngine,
        recorder: TrajectoryRecorder | None = None,
        tenant_id: str = "default",
        task_type: str = "generic",
    ) -> None:
        self.engine = engine
        self.recorder = recorder
        self.tenant_id = tenant_id
        self.task_type = task_type
        self._seq = 0

    # --- LangChain AgentMiddleware surface ---------------------------------

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Gate a model call: DENY blocks, ROUTE overrides the model, and
        ESCALATE/ALLOW pass through. Records exactly one decision event."""
        params = self._params_from_model_request(request)
        response = self._gate(params, tool=None)
        if response.decision is DecisionKind.DENY:
            raise MiddlewareInterrupt(response.rationale)
        if response.decision is DecisionKind.ESCALATE:
            # the decision is recorded; execution continues so the escalation
            # review can carry the model's draft (HITL resolves via review)
            pass
        if (
            response.decision is DecisionKind.ROUTE
            and response.selected_model
            and hasattr(request, "override")
        ):
            request.override(model=response.selected_model)
        return handler(request)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Gate a tool call through the engine (delegation, never duplicated
        policy logic); DENY raises ToolDeniedError so the framework's block
        primitive applies."""
        tool_name = getattr(request, "tool", None) or getattr(request, "name", None)
        args = getattr(request, "args", None) or {}
        response = self._gate(
            {"model": getattr(request, "model", None)}, tool=tool_name, args=args
        )
        if response.decision is DecisionKind.DENY:
            raise ToolDeniedError(tool_name or "?", response.rationale)
        return handler(request)

    # --- legacy hook vocabulary (thin aliases) ------------------------------

    def before_model(self, params: dict[str, Any]) -> dict[str, Any]:
        """Legacy alias: gate a model call; raises MiddlewareInterrupt on DENY."""
        response = self._gate(params, tool=params.get("tool") or _MODEL_ACTION)
        if response.decision is DecisionKind.DENY:
            raise MiddlewareInterrupt(response.rationale)
        return params

    def before_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Legacy alias: gate a tool call; raises ToolDeniedError on DENY."""
        response = self._gate({}, tool=tool_name, args=args)
        if response.decision is DecisionKind.DENY:
            raise ToolDeniedError(tool_name, response.rationale)
        return args

    def after_model(self, response: Any) -> Any:
        """No-op: the decision event is recorded once, in the gate hook.
        A second event here would contradict ESCALATE decisions."""
        return response

    # --- core --------------------------------------------------------------

    def _gate(
        self,
        params: dict[str, Any],
        *,
        tool: str | None,
        args: dict[str, Any] | None = None,
        has_risk: bool | None = None,
    ) -> DecisionResponse:
        request = self.to_request(params, tool=tool, args=args)
        response = self.engine.decide(request)
        self._record(request, response)
        return response

    def _params_from_model_request(self, request: Any) -> dict[str, Any]:
        model = getattr(request, "model", None)
        if isinstance(model, str):
            return {"model": model}
        model_id = getattr(model, "model", None) or getattr(model, "name", None)
        messages = getattr(request, "messages", None) or []
        return {"model": model_id, "messages": messages}

    def to_request(
        self,
        params: dict[str, Any],
        *,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> DecisionRequest:
        """Build a DecisionRequest from hook inputs.

        The proposed action carries ``tool`` (defaulting to ``model_call``
        for model gates so allowlist policies gate model calls too), plus
        model/tools metadata. Messages are never persisted raw — only their
        SHA-256 hash travels into trajectory_state.
        """
        messages = params.get("messages") or []
        last_text = ""
        if messages:
            last = messages[-1]
            last_text = last.get("content", "") if isinstance(last, dict) else str(last)
        return DecisionRequest(
            tenant_id=self.tenant_id,
            task_type=params.get("task_type", self.task_type),
            trajectory_state={
                "last_message_hash": hashlib.sha256(last_text.encode()).hexdigest(),
                **params.get("trajectory_state", {}),
            },
            proposed_action={
                "tool": tool or _MODEL_ACTION,
                "model": params.get("model"),
                "tools": params.get("tools", []),
                "args": dict(args or {}),
            },
            risk_features={
                "calibrated_p": params.get("calibrated_p", 0.0),
            },
            available_models=list(params.get("available_models", [])),
            policy_ref=params.get("policy_ref", ""),
        )

    def _record(self, request: DecisionRequest, response: DecisionResponse) -> None:
        if self.recorder is None:
            return
        from datetime import datetime

        from lossbench.schema import DecisionEvent, Severity

        self._seq += 1
        severity_raw = request.trajectory_state.get("severity", Severity.LOW.value)
        event = DecisionEvent(
            event_id=f"mw-{self.name}-{self._seq}",
            trace_id="unknown",
            trajectory_id="unknown",
            task_id="unknown",
            timestamp=datetime.now(UTC),
            input_snapshot_hash=request.trajectory_state.get("last_message_hash", ""),
            prompt_hash=request.trajectory_state.get("last_message_hash", ""),
            model_id=response.selected_model or request.proposed_action.get("model") or "",
            decision=response.decision,
            rationale=response.rationale,
            policy_id=response.policy_ref,
            cost_model_id="",
            observed_outcome={"severity": severity_raw},
            expected_loss=response.expected_loss,
            calibrated_probability=response.confidence,
            risk_features=dict(request.risk_features),
        )
        self.recorder.record_decision(event)
