"""P1.9 CLI skeleton tests: entrypoint, groups, commands, exit codes."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from click.testing import CliRunner

from lossbench.cli.main import build_group
from lossbench.schema import DecisionKind, DecisionResponse

runner = CliRunner()
cli = build_group()

POLICY_YAML = (
    'id: test-policy\n'
    'revision: "0.1.0"\n'
    "cost_model_id: reconciliation\n"
    "escalation_threshold: 0.5\n"
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _request(risk: float) -> str:
    return json.dumps(
        {
            "tenant_id": "t1",
            "task_type": "reconciliation",
            "trajectory_state": {},
            "proposed_action": {"action": "post"},
            "risk_features": {"calibrated_p": risk},
            "available_models": [],
            "policy_ref": "test-policy",
        },
        sort_keys=True,
    )


def _sim_event(event_id: int, p: float, severity: str, error: bool) -> str:
    record = {
        "event_id": f"e{event_id}",
        "trace_id": "tr1",
        "trajectory_id": "tj1",
        "task_id": "tk1",
        "timestamp": "2026-08-14T00:00:00Z",
        "input_snapshot_hash": "in",
        "prompt_hash": "pr",
        "model_id": "m",
        "decision": "ALLOW",
        "policy_id": "test-policy",
        "cost_model_id": "reconciliation",
        "calibrated_probability": p,
        "severity": severity,
        "error": error,
    }
    return json.dumps(record, sort_keys=True)


def test_version_command():
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", result.output.strip())


def test_costs_list():
    result = runner.invoke(cli, ["costs", "list"])
    assert result.exit_code == 0
    for profile_id in ["flat", "reconciliation", "principal_risk", "review_heavy"]:
        assert profile_id in result.output


def test_metrics_check_known_value():
    line = json.dumps(
        {
            "errors": [True, False, True],
            "severities": ["HIGH", "LOW", "HIGH"],
            "profile_id": "reconciliation",
        }
    )
    result = runner.invoke(cli, ["metrics", "check"], input="\n".join([line] * 3) + "\n")
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["severity_weighted_loss"] == 20.0
    assert "ece" in data
    assert "n_bins" in data


def _invoke_decide(*extra: str):
    return runner.invoke(cli, ["decide", *extra])


def test_decide_valid(tmp_path: Path):
    policy = _write(tmp_path / "policy.yaml", POLICY_YAML)
    request_high = _write(tmp_path / "request_high.json", _request(0.9))
    request_low = _write(tmp_path / "request_low.json", _request(0.1))
    result = _invoke_decide(
        "--request", str(request_high), "--policy", str(policy), "--cost-model", "reconciliation"
    )
    assert result.exit_code == 0
    response = DecisionResponse.model_validate(json.loads(result.output))
    assert response.decision is DecisionKind.ESCALATE
    assert response.requires_human
    result_low = _invoke_decide(
        "--request", str(request_low), "--policy", str(policy), "--cost-model", "reconciliation"
    )
    assert result_low.exit_code == 0
    response_low = DecisionResponse.model_validate(json.loads(result_low.output))
    assert response_low.decision is DecisionKind.ALLOW
    assert not response_low.requires_human


def test_decide_invalid_policy_raises(tmp_path: Path):
    request = _write(tmp_path / "request.json", _request(0.9))
    bad = _write(tmp_path / "policy.yaml", "id: *undefined_alias\n")
    result = _invoke_decide(
        "--request", str(request), "--policy", str(bad), "--cost-model", "reconciliation"
    )
    assert result.exit_code == 1
    assert "policy" in result.output.lower()


def _invoke_simulate(*extra: str):
    return runner.invoke(cli, ["simulate", *extra])


def test_simulate_empty_trace(tmp_path: Path):
    policy = _write(tmp_path / "policy.yaml", POLICY_YAML)
    trace = _write(tmp_path / "trace.jsonl", "")
    result = _invoke_simulate(
        "--trace", str(trace), "--policy", str(policy), "--cost-model", "reconciliation"
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {
        "before": 0.0,
        "after": 0.0,
        "review_load_before": 0.0,
        "review_load_after": 0.0,
    }


def test_simulate_threshold_effect(tmp_path: Path):
    # Policy threshold 0.95 is suboptimal: 5 HIGH p=0.9 errors stay unreviewed
    # (business loss 50). Best threshold 0.5 escalates them (5 reviews, loss 5).
    policy = _write(
        tmp_path / "policy.yaml",
        "id: sim-policy\n"
        "cost_model_id: reconciliation\n"
        "escalation_threshold: 0.95\n"
        "model_tiers: {}\n",
    )
    lines = [_sim_event(i, 0.9, "HIGH", True) for i in range(5)]
    lines += [_sim_event(i, 0.1, "LOW", False) for i in range(5, 10)]
    trace = _write(tmp_path / "trace.jsonl", "\n".join(lines) + "\n")
    result = _invoke_simulate(
        "--trace", str(trace), "--policy", str(policy), "--cost-model", "reconciliation"
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert set(data) == {"before", "after", "review_load_before", "review_load_after"}
    assert data["before"] > data["after"]
    assert data["before"] == 50.0
    assert data["after"] == 5.0
    assert data["review_load_before"] == 0.0
    assert data["review_load_after"] == 0.5


def test_metrics_check_bad_line_raises():
    result = runner.invoke(cli, ["metrics", "check"], input='{"errors": [true\n')
    assert result.exit_code == 1


def test_cli_entrypoint_subprocess():
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["uv", "run", "lossbench", "version"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(r"\d+\.\d+\.\d+", result.stdout)
