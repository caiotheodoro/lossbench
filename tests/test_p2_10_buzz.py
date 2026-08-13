import hashlib

import pytest

from lossbench.buzz import BuzzOutbox, build_payload
from lossbench.hitl.review import ReviewRequest, ReviewResolution
from lossbench.util.canonical import canonical_json


def make_request(decision_id: str = "dec-1", tenant_id: str = "tenant-a") -> ReviewRequest:
    return ReviewRequest(
        decision_id=decision_id,
        trajectory_id=f"traj-{decision_id}",
        tenant_id=tenant_id,
        task_id=f"task-{decision_id}",
        proposed_action={"type": "transfer", "amount": 1000},
        expected_loss=12.5,
        rationale="high risk amount",
        policy_ref="policy-1",
    )


def make_resolution(decision_id: str = "dec-1", resolution: str = "APPROVE") -> ReviewResolution:
    return ReviewResolution(
        decision_id=decision_id,
        reviewer="auditor",
        resolution=resolution,
    )


def test_enqueue_and_pending():
    outbox = BuzzOutbox()
    event = outbox.enqueue_review_request(make_request(), community="com-a")
    pending = outbox.pending()
    assert len(pending) == 1
    entry = pending[0]
    assert entry.outbox_id == event.outbox_id
    assert len(entry.outbox_id) == 16
    assert entry.decision_id == "dec-1"
    assert entry.tenant_id == "tenant-a"
    assert entry.community == "com-a"
    assert entry.kind == "REVIEW_REQUEST"
    assert entry.payload["kind"] == "REVIEW_REQUEST"
    assert entry.payload["decision_id"] == "dec-1"
    assert entry.payload["tenant_id"] == "tenant-a"
    assert entry.payload["community"] == "com-a"
    expected_hash = hashlib.sha256(canonical_json(entry.payload["extra"]).encode()).hexdigest()
    assert entry.payload["payload_hash"] == expected_hash


def test_idempotent_per_decision():
    outbox = BuzzOutbox()
    first = outbox.enqueue_review_request(make_request(), community="com-a")
    second = outbox.enqueue_review_request(make_request(), community="com-b")
    assert second.outbox_id == first.outbox_id
    assert len(outbox.pending()) == 1
    assert outbox.pending()[0].community == "com-a"


def test_mark_published():
    outbox = BuzzOutbox()
    event = outbox.enqueue_review_request(make_request(), community="com-a")
    outbox.mark_published(event.outbox_id)
    assert outbox.pending() == []
    published = outbox.published()
    assert len(published) == 1
    assert published[0].outbox_id == event.outbox_id


def test_mark_published_unknown_raises():
    outbox = BuzzOutbox()
    with pytest.raises(ValueError):
        outbox.mark_published("0000000000000000")


def test_resolution_callback_accepted():
    outbox = BuzzOutbox()
    request = outbox.enqueue_review_request(make_request(), community="com-a")
    outbox.mark_published(request.outbox_id)
    resolution = outbox.enqueue_resolution(make_resolution(), community="com-a")
    result = outbox.resolve_callback(
        {"decision_id": "dec-1", "resolution": "APPROVE", "reviewer": "auditor"}
    )
    assert result == {"accepted": True, "outbox_id": resolution.outbox_id}
    published_ids = {e.outbox_id for e in outbox.published()}
    assert resolution.outbox_id in published_ids


def test_resolution_callback_rejects_unpublished_request():
    outbox = BuzzOutbox()
    outbox.enqueue_review_request(make_request(), community="com-a")
    outbox.enqueue_resolution(make_resolution(), community="com-a")
    with pytest.raises(ValueError, match="not published"):
        outbox.resolve_callback(
            {"decision_id": "dec-1", "resolution": "APPROVE", "reviewer": "auditor"}
        )


def test_resolution_callback_rejects_unknown_request():
    outbox = BuzzOutbox()
    with pytest.raises(ValueError, match="no outbox review request"):
        outbox.resolve_callback(
            {"decision_id": "missing", "resolution": "APPROVE", "reviewer": "auditor"}
        )


def test_resolution_callback_rejects_bad_resolution():
    outbox = BuzzOutbox()
    with pytest.raises(ValueError, match="invalid resolution"):
        outbox.resolve_callback(
            {"decision_id": "dec-1", "resolution": "MAYBE", "reviewer": "auditor"}
        )


def test_event_for():
    outbox = BuzzOutbox()
    outbox.enqueue_review_request(make_request(), community="com-a")
    resolution = outbox.enqueue_resolution(make_resolution(), community="com-a")
    found = outbox.event_for("dec-1", "REVIEW_RESOLVED")
    assert found is not None
    assert found.outbox_id == resolution.outbox_id
    assert found.kind == "REVIEW_RESOLVED"
    assert outbox.event_for("dec-1", "REVIEW_REQUEST").kind == "REVIEW_REQUEST"
    assert outbox.event_for("missing", "REVIEW_REQUEST") is None


def test_build_payload_deterministic():
    extra_a = {"b": 2, "a": {"nested": 1}, "list": [3, 1, 2]}
    extra_b = {"a": {"nested": 1}, "list": [3, 1, 2], "b": 2}
    payload_a = build_payload(None, kind="REVIEW_REQUEST", decision_id="dec-1",
                              tenant_id="tenant-a", extra=extra_a)
    payload_b = build_payload(None, kind="REVIEW_REQUEST", decision_id="dec-1",
                              tenant_id="tenant-a", extra=extra_b)
    assert payload_a == payload_b
    assert payload_a["kind"] == "REVIEW_REQUEST"
    assert payload_a["decision_id"] == "dec-1"
    assert payload_a["tenant_id"] == "tenant-a"
    assert payload_a["payload_hash"] == hashlib.sha256(
        canonical_json(extra_a).encode()
    ).hexdigest()


def test_persistence(tmp_path):
    path = str(tmp_path / "outbox.duckdb")
    outbox = BuzzOutbox(path)
    outbox.enqueue_review_request(make_request(), community="com-a")
    reopened = BuzzOutbox(path)
    pending = reopened.pending()
    assert len(pending) == 1
    assert pending[0].decision_id == "dec-1"
    assert pending[0].community == "com-a"
