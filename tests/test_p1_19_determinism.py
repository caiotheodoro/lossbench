from datetime import UTC, datetime

from lossbench.schema import DecisionEvent, DecisionKind
from lossbench.util import (
    assert_identical,
    canonical_json,
    freeze_list,
    hash_event,
    seed_rng,
    sha256_hex,
)

ABC_DIGEST = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _event(timestamp: datetime, *, event_id: str = "e1") -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        trace_id="t1",
        trajectory_id="traj-1",
        task_id="task-1",
        timestamp=timestamp,
        input_snapshot_hash="in-1",
        prompt_hash="pr-1",
        model_id="m1",
        decision=DecisionKind.ALLOW,
        policy_id="p1",
        cost_model_id="c1",
        risk_features={"a": 0.1, "b": 0.9},
        created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    )


def test_canonical_order_insensitive():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_nested():
    left = {"z": [3, 2], "b": {"y": {"k": 1, "j": (2, 3)}, "x": [{"m": 1}]}, "a": None}
    right = {"a": None, "b": {"x": [{"m": 1}], "y": {"j": (2, 3), "k": 1}}, "z": [3, 2]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_json({"t": (1, 2)}) == canonical_json({"t": [1, 2]})


def test_canonical_numbers():
    assert canonical_json({"x": 1.0}) != canonical_json({"x": 1})
    first = canonical_json({"n": float("nan")})
    assert first == canonical_json({"n": float("nan")})
    assert "NaN" in first


def test_canonical_deterministic():
    obj = {"d": {"b": 1.5, "a": [True, None, {"z": "x"}]}, "n": 3}
    results = {canonical_json(obj) for _ in range(10)}
    assert len(results) == 1


def test_sha256_hex_known():
    assert sha256_hex(b"abc") == ABC_DIGEST
    assert sha256_hex("abc") == ABC_DIGEST


def test_hash_event_stable_and_excludes():
    ts1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    ts2 = datetime(2026, 6, 1, 8, 30, 0, tzinfo=UTC)
    assert hash_event(_event(ts1)) == hash_event(_event(ts1))
    assert hash_event(_event(ts1)) != hash_event(_event(ts2))
    assert hash_event(_event(ts1), exclude=frozenset({"timestamp"})) == hash_event(
        _event(ts2), exclude=frozenset({"timestamp"})
    )


def test_freeze_list_membership():
    frozen = freeze_list([{"a": 1, "b": 2}, [1, 2], {"b": 2, "a": 1}])
    assert len(frozen) == 2
    assert canonical_json({"b": 2, "a": 1}) in frozen
    assert canonical_json([1, 2]) in frozen


def test_assert_identical_raises_with_context():
    assert_identical("a", "a")
    try:
        assert_identical("a", "b", context="determinism gate")
    except AssertionError as exc:
        assert "determinism gate" in str(exc)
    else:
        raise AssertionError("expected AssertionError")


def test_seed_rng_deterministic():
    rng_a = seed_rng(42)
    rng_b = seed_rng(42)
    assert [rng_a.random() for _ in range(10)] == [rng_b.random() for _ in range(10)]
    assert seed_rng(1).random() != seed_rng(2).random()
