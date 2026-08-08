import json
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from lossbench.ledger import AuditLedger
from lossbench.schema import DecisionEvent, DecisionKind


def make_event(
    event_id: str, trajectory_id: str = "traj-1", tenant_id: str = "tenant-a"
) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        tenant_id=tenant_id,
        trace_id=f"trace-{event_id}",
        trajectory_id=trajectory_id,
        task_id=f"task-{event_id}",
        timestamp=datetime(2026, 8, 14, 12, 0, 0),
        input_snapshot_hash="in-1",
        prompt_hash="prompt-1",
        model_id="model-a",
        decision=DecisionKind.ALLOW,
        policy_id="policy-1",
        cost_model_id="cost-1",
    )


def test_append_and_get_roundtrip():
    ledger = AuditLedger()
    event = make_event("ev-1")
    ledger.append(event)
    assert ledger.get("ev-1") == event
    ledger.close()


def test_immutable_id():
    ledger = AuditLedger()
    event = make_event("ev-dup")
    ledger.append(event)
    with pytest.raises(ValueError):
        ledger.append(event)
    assert ledger.get("ev-dup") == event
    assert ledger.verify()["n_events"] == 1
    ledger.close()


def test_chain_consistency():
    ledger = AuditLedger()
    hashes = [ledger.append(make_event(f"ev-{i}")) for i in range(1, 4)]
    result = ledger.verify()
    assert result["valid"] is True
    assert result["n_events"] == 3
    assert result["head"] == hashes[-1]
    assert result["first_bad_seq"] is None
    ledger.close()


def test_chain_detects_tamper(tmp_path: Path):
    path = str(tmp_path / "ledger.duckdb")
    ledger = AuditLedger(path)
    ledger.append(make_event("ev-1"))
    ledger.append(make_event("ev-2"))
    ledger.append(make_event("ev-3"))
    ledger.close()
    raw = duckdb.connect(path)
    raw.execute(
        "UPDATE events SET event_json = ? WHERE event_id = ?", ["tampered", "ev-2"]
    )
    raw.close()
    reopened = AuditLedger(path)
    result = reopened.verify()
    assert result["valid"] is False
    assert result["first_bad_seq"] == 2
    reopened.close()


def test_trajectory_and_tenant_filters():
    ledger = AuditLedger()
    ledger.append(make_event("ev-1", trajectory_id="traj-a", tenant_id="tenant-a"))
    ledger.append(make_event("ev-2", trajectory_id="traj-b", tenant_id="tenant-a"))
    ledger.append(make_event("ev-3", trajectory_id="traj-a", tenant_id="tenant-b"))
    assert [e.event_id for e in ledger.events_by_trajectory("traj-a")] == [
        "ev-1",
        "ev-3",
    ]
    assert [e.event_id for e in ledger.events_by_tenant("tenant-a")] == [
        "ev-2",
        "ev-1",
    ]
    assert [e.event_id for e in ledger.events_by_tenant("tenant-a", limit=1)] == [
        "ev-2",
    ]
    ledger.close()


def test_export_jsonl_and_reload(tmp_path: Path):
    ledger = AuditLedger()
    events = [
        make_event("ev-1", tenant_id="tenant-a"),
        make_event("ev-2", tenant_id="tenant-b"),
        make_event("ev-3", tenant_id="tenant-a"),
    ]
    for event in events:
        ledger.append(event)
    out = tmp_path / "ledger.jsonl"
    assert ledger.export_jsonl(out) == 3
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    reloaded = [DecisionEvent.model_validate(line["event"]) for line in lines]
    assert reloaded == events
    assert [line["seq"] for line in lines] == [1, 2, 3]
    assert [line["prev_hash"] for line in lines[1:]] == [
        line["chain_hash"] for line in lines[:-1]
    ]
    out_tenant = tmp_path / "tenant-a.jsonl"
    assert ledger.export_jsonl(out_tenant, tenant_id="tenant-a") == 2
    tenant_lines = out_tenant.read_text().splitlines()
    assert [
        DecisionEvent.model_validate(json.loads(line)["event"]).tenant_id
        for line in tenant_lines
    ] == ["tenant-a", "tenant-a"]
    ledger.close()


def test_deterministic_hashes():
    events = [make_event(f"ev-{i}") for i in range(1, 4)]
    first = AuditLedger()
    second = AuditLedger()
    assert [first.append(e) for e in events] == [second.append(e) for e in events]
    assert first.verify() == second.verify()
    first.close()
    second.close()


def test_empty_ledger_verify():
    ledger = AuditLedger()
    result = ledger.verify()
    assert result["valid"] is True
    assert result["n_events"] == 0
    assert result["first_bad_seq"] is None
    ledger.close()
