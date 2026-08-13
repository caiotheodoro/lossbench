"""DeepSeek Harness (dsh) plugin surface over the LossBench policy core.

Defines the Python-side dsh plugin contract: a manifest template, serializable
hook payloads, and the bridge that maps the dsh hook vocabulary (beforeModel,
beforeTool, onResolution) onto the same deterministic policy engine used by the
rest of LossBench. No Node runtime is imported; a thin JS shim (shipped in a
later packaging step) maps these JSON shapes onto dsh behavior.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from lossbench.policy import PolicyEngine
from lossbench.record import TrajectoryRecorder
from lossbench.schema import DecisionEvent, DecisionKind, DecisionRequest, Severity

_HARNESS_ID = "dsh-plugin"
_HARNESS_REVISION = "0.1.0"
_MODEL_ACTION = "model_call"
_RESOLUTIONS = ("APPROVE", "REJECT", "AMEND")
_METADATA_KEYS = frozenset(
    {"severity", "trajectory_id", "task_id", "model_id", "tenant_id", "action", "available_models"}
)


def build_manifest(
    plugin_id: str,
    version: str,
    hooks: list[str],
    bridge_url: str,
    auth_header: str = "x-lossbench-key",
) -> dict[str, Any]:
    """Return a dsh plugin manifest template.

    Shape: {"id", "version", "hooks", "bridge"} where bridge is {"url",
    "auth_header", "timeout_s": 30}. The manifest declares the plugin
    identity a dsh registry (topic dsh-plugin) uses to load the shim, the
    hook names the shim registers, and the HTTP endpoint the shim calls
    back into, with the header carrying the shared key and a request
    timeout in seconds.
    """
    return {
        "id": plugin_id,
        "version": version,
        "hooks": list(hooks),
        "bridge": {
            "url": bridge_url,
            "auth_header": auth_header,
            "timeout_s": 30,
        },
    }


def hook_payload(
    hook: str,
    event: DecisionEvent | None = None,
    tool_name: str | None = None,
    args: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize one dsh hook call into the JSON body the shim posts.

    Keys are exactly {hook, event, tool_name, args, messages}; absent
    inputs serialize to None and a DecisionEvent serializes via
    model_dump(mode="json") so the payload is JSON-safe end to end.
    """
    return {
        "hook": hook,
        "event": event.model_dump(mode="json") if event is not None else None,
        "tool_name": tool_name,
        "args": args,
        "messages": messages,
    }


def manifest_json(manifest: dict[str, Any]) -> str:
    """Canonical JSON for a manifest: keys sorted recursively, 2-space indent."""
    return json.dumps(manifest, sort_keys=True, indent=2)


class ToolDeniedError(RuntimeError):
    """Bridge-side representation of a tool denial from the policy rules.

    Carries the offending tool name and the engine rationale; the JS shim
    maps the block envelope carrying these fields to dsh's own
    ToolDeniedError behavior.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(reason)
        self.tool_name = tool_name
        self.reason = reason


class DshPluginBridge:
    """Adapts the policy engine to the dsh hook vocabulary.

    beforeModel and beforeTool build DecisionRequests from the hook inputs,
    apply the engine, and map outcomes to the JSON envelope the JS shim
    translates into dsh behavior: continue / block / escalate. Every
    outcome is recorded as a DecisionEvent when a recorder is attached.

    Hook inputs may carry optional lossbench metadata to drive the policy:
    a "lossbench" dict on any message (beforeModel) or top-level keys in
    tool args (beforeTool). Recognized keys: calibrated_p, severity,
    trajectory_id, task_id, model_id, tenant_id, action (beforeModel only),
    available_models (beforeModel only). Other numeric keys merge into
    risk_features.
    """

    def __init__(self, engine: PolicyEngine, recorder: TrajectoryRecorder | None = None) -> None:
        """Bind the bridge to a policy engine and an optional recorder."""
        self._engine = engine
        self._recorder = recorder

    def on_before_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Decision envelope for dsh's beforeModel hook.

        Returns {"action": "continue", "payload": {...}} for ALLOW and
        ROUTE, {"action": "block", "reason": ...} for DENY, and
        {"action": "escalate", "reason": ...} for ESCALATE. The model call
        is treated as an action named "model_call" (overridable via
        lossbench metadata key "action") so allowlist/deny rules apply
        uniformly; the continue payload carries the DecisionResponse and
        the recorded DecisionEvent.
        """
        metadata = self._merge_message_metadata(messages)
        request = DecisionRequest(
            tenant_id=metadata.get("tenant_id") or "default",
            task_type="model",
            trajectory_state={"severity": metadata.get("severity", Severity.LOW.value)},
            proposed_action={"tool": metadata.get("action") or _MODEL_ACTION},
            risk_features=self._risk_features(metadata),
            available_models=self._available_models(metadata),
            policy_ref=self._engine.bundle.id,
        )
        response = self._engine.decide(request)
        return self._envelope(response, source=messages, metadata=metadata, request=request)

    def on_before_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Decision envelope for dsh's beforeTool hook.

        Same envelope as on_before_model; DENY from the engine's
        allowlist/deny rules returns a block envelope whose reason is
        carried by a ToolDeniedError. Tool args may carry calibrated_p,
        severity, and trajectory context at the top level.
        """
        metadata = dict(args or {})
        request = DecisionRequest(
            tenant_id=metadata.get("tenant_id") or "default",
            task_type="tool",
            trajectory_state={"severity": metadata.get("severity", Severity.LOW.value)},
            proposed_action={"tool": tool_name, "args": dict(args or {})},
            risk_features=self._risk_features(metadata),
            available_models=[],
            policy_ref=self._engine.bundle.id,
        )
        response = self._engine.decide(request)
        return self._envelope(
            response,
            source={"tool_name": tool_name, "args": args or {}},
            metadata=metadata,
            request=request,
            tool_name=tool_name,
        )

    def on_resolution(self, resolution: dict[str, Any]) -> dict[str, Any]:
        """Record a human resolution payload {decision_id, resolution, reviewer}.

        resolution must be APPROVE, REJECT, or AMEND. APPROVE maps to an
        ALLOW event and REJECT to DENY; AMEND requires an amended_action
        mapping and records a VERIFY of the amended action as proposed
        action (matching ReviewService's AMEND->VERIFY semantics). Anything
        else raises ValueError. The resolution event's id is derived as
        '{decision_id}@resolved' so it can never collide with the source
        decision's ledger id. The returned dict is the serialized
        DecisionEvent appended via the recorder.
        """
        outcome, decision_id, reviewer, amended = self._validate_resolution(resolution)
        if outcome == "REJECT":
            decision = DecisionKind.DENY
            proposed_action = None
        elif outcome == "AMEND":
            decision = DecisionKind.VERIFY
            proposed_action = amended
        else:
            decision = DecisionKind.ALLOW
            proposed_action = None
        event = self._record(
            decision=decision,
            rationale=f"human {outcome.lower()} by {reviewer}",
            source=resolution,
            metadata=resolution,
            event_id=f"{decision_id}@resolved",
            proposed_action=proposed_action,
            observed_outcome={"resolution": outcome, "reviewer": reviewer},
        )
        return event.model_dump(mode="json")

    def _envelope(
        self,
        response,
        *,
        source: Any,
        metadata: dict[str, Any],
        request: DecisionRequest,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        if response.decision is DecisionKind.DENY:
            if tool_name is not None:
                error = ToolDeniedError(tool_name, response.rationale)
                reason = error.reason
            else:
                reason = response.rationale
            self._record(
                decision=DecisionKind.DENY,
                rationale=reason,
                source=source,
                metadata=metadata,
                tool_name=tool_name,
                proposed_action=request.proposed_action,
            )
            return {"action": "block", "reason": reason}
        if response.decision is DecisionKind.ESCALATE:
            self._record(
                decision=DecisionKind.ESCALATE,
                rationale=response.rationale,
                source=source,
                metadata=metadata,
                tool_name=tool_name,
                proposed_action=request.proposed_action,
                expected_loss=response.expected_loss,
            )
            return {"action": "escalate", "reason": response.rationale}
        event = self._record(
            decision=response.decision,
            rationale=response.rationale,
            source=source,
            metadata=metadata,
            tool_name=tool_name,
            proposed_action=request.proposed_action,
            expected_loss=response.expected_loss,
        )
        return {
            "action": "continue",
            "payload": {
                "decision": response.model_dump(mode="json"),
                "event": event.model_dump(mode="json"),
            },
        }

    def _record(
        self,
        *,
        decision: DecisionKind,
        rationale: str,
        source: Any,
        metadata: dict[str, Any],
        event_id: str | None = None,
        tool_name: str | None = None,
        proposed_action: dict[str, Any] | None = None,
        observed_outcome: dict[str, Any] | None = None,
        expected_loss: float | None = None,
    ) -> DecisionEvent:
        fingerprint = self._fingerprint(source)
        trajectory_id = metadata.get("trajectory_id") or fingerprint
        generated_id = event_id or uuid.uuid4().hex
        calibrated = metadata.get("calibrated_p")
        calibrated_probability = (
            float(calibrated)
            if isinstance(calibrated, (int, float)) and not isinstance(calibrated, bool)
            else None
        )
        event = DecisionEvent(
            event_id=generated_id,
            tenant_id=metadata.get("tenant_id") or "default",
            trace_id=hashlib.sha256(f"{generated_id}:{trajectory_id}".encode()).hexdigest(),
            trajectory_id=trajectory_id,
            task_id=metadata.get("task_id") or fingerprint[:16],
            timestamp=datetime.now(UTC),
            input_snapshot_hash=fingerprint,
            prompt_hash=fingerprint,
            model_id=metadata.get("model_id") or "",
            harness_id=_HARNESS_ID,
            harness_revision=_HARNESS_REVISION,
            tool_name=tool_name,
            proposed_action=proposed_action,
            observed_outcome=observed_outcome,
            risk_features=self._risk_features(metadata),
            calibrated_probability=calibrated_probability,
            expected_loss=expected_loss,
            decision=decision,
            rationale=rationale,
            policy_id=self._engine.bundle.id,
            policy_revision=self._engine.bundle.revision,
            cost_model_id=self._engine.cost_profile.id,
        )
        if self._recorder is not None:
            self._recorder.record_decision(event)
        return event

    def _validate_resolution(
        self, resolution: dict[str, Any]
    ) -> tuple[str, str, str, dict[str, Any] | None]:
        if not isinstance(resolution, dict):
            raise ValueError(
                "resolution must be a mapping with decision_id, resolution, and reviewer"
            )
        decision_id = resolution.get("decision_id")
        outcome = resolution.get("resolution")
        reviewer = resolution.get("reviewer")
        if not decision_id or not outcome or not reviewer:
            raise ValueError("resolution payload requires decision_id, resolution, and reviewer")
        if outcome not in _RESOLUTIONS:
            raise ValueError(f"resolution must be one of APPROVE, REJECT, AMEND; got {outcome!r}")
        amended = resolution.get("amended_action")
        if outcome == "AMEND" and not isinstance(amended, dict):
            raise ValueError("AMEND resolution requires an amended_action mapping")
        return outcome, decision_id, reviewer, amended

    def _merge_message_metadata(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for message in messages or []:
            if isinstance(message, dict):
                tagged = message.get("lossbench")
                if isinstance(tagged, dict):
                    metadata.update(tagged)
        return metadata

    def _risk_features(self, metadata: dict[str, Any]) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in metadata.items()
            if key not in _METADATA_KEYS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

    def _available_models(self, metadata: dict[str, Any]) -> list[str]:
        models = metadata.get("available_models", [])
        if not isinstance(models, list):
            return []
        return [model for model in models if isinstance(model, str)]

    def _fingerprint(self, source: Any) -> str:
        return hashlib.sha256(
            json.dumps(source, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
