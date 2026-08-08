"""Tests for the P1.8 trajectory recorder and proxy mode."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from lossbench.record import TrajectoryRecorder, event_from_trace, run_proxy
from lossbench.schema import DecisionEvent, DecisionKind


def _build_event(
    trajectory_id: str,
    task_id: str,
    event_id: str = "evt-1",
    decision: DecisionKind = DecisionKind.ALLOW,
) -> DecisionEvent:
    """Build a minimal valid DecisionEvent for tests."""
    return DecisionEvent(
        event_id=event_id,
        trace_id="trace-1",
        trajectory_id=trajectory_id,
        task_id=task_id,
        timestamp=datetime.now(UTC),
        input_snapshot_hash="in-1",
        prompt_hash="pr-1",
        model_id="m1",
        calibrated_probability=0.7,
        expected_loss=2.5,
        decision=decision,
        policy_id="p1",
        cost_model_id="flat",
    )


def test_record_and_retrieve() -> None:
    """Two recorded events are returned in record order."""
    recorder = TrajectoryRecorder()
    recorder.record_decision(_build_event("traj-1", "task-1", event_id="a"))
    recorder.record_decision(_build_event("traj-1", "task-1", event_id="b"))
    events = recorder.trajectory_events("traj-1")
    assert [e.event_id for e in events] == ["a", "b"]


def test_flush_clears_buffer() -> None:
    """Flush returns the buffered events and empties the buffer."""
    recorder = TrajectoryRecorder()
    recorder.record_decision(_build_event("traj-1", "task-1"))
    flushed = recorder.flush()
    assert [e.event_id for e in flushed] == ["evt-1"]
    assert recorder.flush() == []


def test_otel_span_emitted() -> None:
    """Recording an event emits a span named lossbench.decision."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = otel_trace.get_tracer_provider()
    otel_trace.set_tracer_provider(provider)
    try:
        recorder = TrajectoryRecorder()
        event = _build_event("traj-otel", "task-otel")
        recorder.record_decision(event)
        spans = [s for s in exporter.get_finished_spans() if s.name == "lossbench.decision"]
        assert len(spans) == 1
        attributes = spans[0].attributes
        assert attributes["event_id"] == event.event_id
        assert attributes["decision"] == event.decision.value
        assert attributes["model_id"] == event.model_id
        assert attributes["policy_id"] == event.policy_id
        assert attributes["calibrated_probability"] == event.calibrated_probability
        assert attributes["expected_loss"] == event.expected_loss
    finally:
        otel_trace.set_tracer_provider(previous)


def test_event_from_trace_roundtrip() -> None:
    """Span attributes reconstructed from a recorded event round-trip it."""
    recorder = TrajectoryRecorder()
    original = _build_event("traj-rt", "task-rt", event_id="evt-rt")
    recorder.record_decision(original)
    span_attrs = {
        key: original.model_dump()[key]
        for key in (
            "event_id",
            "decision",
            "model_id",
            "calibrated_probability",
            "expected_loss",
            "trajectory_id",
            "task_id",
            "input_snapshot_hash",
            "prompt_hash",
        )
    }
    rebuilt = event_from_trace(span_attrs, policy_id="p1", cost_model_id="flat")
    assert rebuilt.event_id == original.event_id
    assert rebuilt.decision == original.decision
    assert rebuilt.model_id == original.model_id
    assert rebuilt.trajectory_id == original.trajectory_id
    assert rebuilt.task_id == original.task_id
    assert rebuilt.input_snapshot_hash == original.input_snapshot_hash
    assert rebuilt.prompt_hash == original.prompt_hash
    assert rebuilt.policy_id == original.policy_id
    assert rebuilt.cost_model_id == original.cost_model_id


def test_event_from_trace_missing_required_raises() -> None:
    """A span attribute dict missing a required key raises ValueError."""
    with pytest.raises(ValueError, match="event_id"):
        event_from_trace(
            {"decision": "ALLOW", "model_id": "m1", "trajectory_id": "t", "task_id": "k"},
            policy_id="p1",
            cost_model_id="flat",
        )


def test_proxy_hermetic(monkeypatch) -> None:
    """run_proxy records an ALLOW event and round-trips the response text."""
    from lossbench.record import proxy as proxy_module

    class FakeMessage:
        content = "42"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["messages"][0]["content"] == "reconcile account 12345"
            assert kwargs["timeout"] == 60
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, base_url, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr(proxy_module, "OpenAI", FakeOpenAI)
    recorder = TrajectoryRecorder()
    prompt = "reconcile account 12345"
    event_dump, text = run_proxy(
        prompt,
        base_url="http://localhost:8000/v1",
        model_id="test-model",
        api_key="sk-test",
        recorder=recorder,
        trajectory_id="traj-p",
        task_id="task-p",
        policy_id="p1",
        cost_model_id="flat",
    )
    assert text == "42"
    assert event_dump["decision"] == "ALLOW"
    expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert event_dump["prompt_hash"] == expected_hash
    assert event_dump["input_snapshot_hash"] == expected_hash
    recorded = recorder.trajectory_events("traj-p")
    assert len(recorded) == 1
    assert recorded[0].decision is DecisionKind.ALLOW


def test_proxy_failure_records_abstain(monkeypatch) -> None:
    """A failing API call records an ABSTAIN decision."""
    from lossbench.record import proxy as proxy_module

    class RaisingCompletions:
        def create(self, **kwargs):
            raise TimeoutError("simulated timeout")

    class RaisingChat:
        completions = RaisingCompletions()

    class FailingOpenAI:
        def __init__(self, base_url, api_key):
            self.chat = RaisingChat()

    monkeypatch.setattr(proxy_module, "OpenAI", FailingOpenAI)
    recorder = TrajectoryRecorder()
    event_dump, text = run_proxy(
        "reconcile account 12345",
        base_url="http://localhost:8000/v1",
        model_id="test-model",
        api_key="sk-test",
        recorder=recorder,
        trajectory_id="traj-f",
        task_id="task-f",
        policy_id="p1",
        cost_model_id="flat",
    )
    assert event_dump["decision"] == "ABSTAIN"
    assert text == ""
    assert recorder.flush()[0].decision is DecisionKind.ABSTAIN


def test_trajectory_isolation() -> None:
    """Events in one trajectory are invisible to another."""
    recorder = TrajectoryRecorder()
    recorder.record_decision(_build_event("traj-a", "task-a", event_id="a1"))
    recorder.record_decision(_build_event("traj-a", "task-a", event_id="a2"))
    recorder.record_decision(_build_event("traj-b", "task-b", event_id="b1"))
    assert [e.event_id for e in recorder.trajectory_events("traj-a")] == ["a1", "a2"]
    assert [e.event_id for e in recorder.trajectory_events("traj-b")] == ["b1"]
