"""Trajectory recording: in-memory DecisionEvent buffers with OTel span emission."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from lossbench.schema import DecisionEvent, DecisionKind

_SPAN_NAME = "lossbench.decision"

_tracer = trace.get_tracer(__name__)


class TrajectoryRecorder:
    """In-memory recorder of DecisionEvents that also emits OTel spans.

    The OpenTelemetry tracer falls back to a no-op when no SDK exporter is
    configured, so the recorder never requires an OTLP endpoint.
    """

    def __init__(self, tenant_id: str = "default"):
        """Create a recorder for the given tenant."""
        self._tenant_id = tenant_id
        self._events: list[DecisionEvent] = []
        self._trajectory_id: str | None = None
        self._task_id: str | None = None

    def start_trajectory(self, trajectory_id: str, task_id: str) -> None:
        """Remember the current trajectory context for subsequent decisions."""
        self._trajectory_id = trajectory_id
        self._task_id = task_id

    def record_decision(self, event: DecisionEvent) -> None:
        """Append to the in-memory buffer and emit a 'lossbench.decision' span.

        The span carries event_id, decision, model_id, policy_id,
        calibrated_probability, and expected_loss as attributes.
        """
        event.created_at = datetime.now(UTC)
        self._events.append(event)
        attributes: dict[str, Any] = {
            "event_id": event.event_id,
            "decision": event.decision.value,
            "model_id": event.model_id,
            "policy_id": event.policy_id,
        }
        if event.calibrated_probability is not None:
            attributes["calibrated_probability"] = event.calibrated_probability
        if event.expected_loss is not None:
            attributes["expected_loss"] = event.expected_loss
        span = _tracer.start_span(_SPAN_NAME, attributes=attributes)
        span.end()

    def trajectory_events(self, trajectory_id: str) -> list[DecisionEvent]:
        """Return events recorded for a trajectory, in record order."""
        return [event for event in self._events if event.trajectory_id == trajectory_id]

    def flush(self) -> list[DecisionEvent]:
        """Return and clear the buffered events."""
        events = self._events
        self._events = []
        return events

    def close(self) -> None:
        """Drop all buffered events."""
        self.flush()


def event_from_trace(
    span_attrs: dict[str, Any],
    *,
    tenant_id: str = "default",
    policy_id: str,
    cost_model_id: str,
) -> DecisionEvent:
    """Rebuild a DecisionEvent from OTel span attributes captured by a third-party tracer.

    Reads event_id, decision, model_id, calibrated_probability, expected_loss,
    trajectory_id, task_id, input_snapshot_hash, and prompt_hash. Raises
    ValueError when a required attribute is missing.
    """
    required = (
        "event_id",
        "decision",
        "model_id",
        "trajectory_id",
        "task_id",
        "input_snapshot_hash",
        "prompt_hash",
    )
    missing = [key for key in required if key not in span_attrs]
    if missing:
        raise ValueError(f"missing required span attributes: {', '.join(missing)}")
    trace_id = hashlib.sha256(
        f"{span_attrs['event_id']}:{span_attrs['trajectory_id']}".encode()
    ).hexdigest()
    return DecisionEvent(
        event_id=str(span_attrs["event_id"]),
        tenant_id=tenant_id,
        trace_id=trace_id,
        trajectory_id=str(span_attrs["trajectory_id"]),
        task_id=str(span_attrs["task_id"]),
        timestamp=datetime.now(UTC),
        input_snapshot_hash=str(span_attrs["input_snapshot_hash"]),
        prompt_hash=str(span_attrs["prompt_hash"]),
        model_id=str(span_attrs["model_id"]),
        calibrated_probability=span_attrs.get("calibrated_probability"),
        expected_loss=span_attrs.get("expected_loss"),
        decision=DecisionKind(str(span_attrs["decision"])),
        policy_id=policy_id,
        cost_model_id=cost_model_id,
    )
