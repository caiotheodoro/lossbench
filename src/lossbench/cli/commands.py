"""Click commands for the lossbench CLI (P1.9 skeleton).

Every command reads or writes JSON/YAML through the pydantic contract models;
all JSON output is emitted with sorted keys for deterministic ordering.
Validation failures surface as click.BadParameter with exit code 1.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import ValidationError

from lossbench.costs.registry import list_cost_profiles, load_cost_profile
from lossbench.metrics.calibration import ece
from lossbench.metrics.loss import severity_weighted_loss
from lossbench.policy.engine import PolicyEngine
from lossbench.policy.fit import fit_escalation_threshold
from lossbench.replay.simulator import ReplayLab
from lossbench.schema import (
    CostProfile,
    DecisionEvent,
    DecisionRequest,
    DecisionResponse,
    PolicyBundle,
    Severity,
)


def _exit_on_error(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except click.ClickException as exc:
            click.echo(f"Error: {exc.format_message()}", err=True)
            raise SystemExit(1) from None
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from None

    return wrapper


@click.group(name="metrics")
def metrics() -> None:
    """Compute stream metrics over recorded decisions."""


@metrics.command(name="check")
@_exit_on_error
def metrics_check() -> None:
    """Read JSON lines of {errors, severities, profile_id} from stdin.

    Prints the mean severity-weighted loss per line plus an ECE summary as
    JSON. ECE uses per-line confidences/correct when present, otherwise a
    constant-confidence baseline (confidence 1.0, correct = not error).
    """
    records = _read_metrics_lines()
    per_line_loss: list[float] = []
    confidences: list[float] = []
    correct: list[bool] = []
    for line_no, record in enumerate(records, start=1):
        try:
            errors = record["errors"]
            severities = [Severity(sev) for sev in record["severities"]]
            profile = load_cost_profile(record["profile_id"])
            per_line_loss.append(severity_weighted_loss(errors, severities, profile))
        except (KeyError, TypeError, ValueError) as exc:
            raise click.BadParameter(f"invalid metrics line {line_no}: {exc}") from exc
        confidences.extend(record.get("confidences", [1.0] * len(errors)))
        correct.extend(record.get("correct", [not err for err in errors]))
    calibration = ece(confidences, correct)
    mean_loss = sum(per_line_loss) / len(per_line_loss) if per_line_loss else 0.0
    summary = {
        "severity_weighted_loss": mean_loss,
        "n_cases": len(records),
        "ece": calibration["ece"],
        "n_bins": calibration["n_bins"],
    }
    click.echo(json.dumps(summary, sort_keys=True))


def _read_metrics_lines() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(
        click.get_text_stream("stdin").read().splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise click.BadParameter(f"invalid JSON on metrics line {line_no}: {exc}") from exc
    return records


@click.group(name="costs")
def costs() -> None:
    """Inspect the shipped cost profiles."""


@costs.command(name="list")
@_exit_on_error
def costs_list() -> None:
    """Print available cost profile ids, one per line."""
    for profile_id in list_cost_profiles():
        click.echo(profile_id)


@click.command(name="decide")
@click.option(
    "--request",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="DecisionRequest JSON file",
)
@click.option(
    "--policy",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="PolicyBundle YAML file",
)
@click.option("--cost-model", required=True, help="Cost profile id (see 'costs list')")
@_exit_on_error
def decide(request: Path, policy: Path, cost_model: str) -> None:
    """Decide a single request under a policy; print the DecisionResponse as JSON."""
    request_model = _load_request(request)
    policy_model = _load_policy(policy)
    profile = _load_profile(cost_model)
    response = _policy_decision(request_model, policy_model, profile)
    click.echo(json.dumps(response.model_dump(mode="json"), sort_keys=True))


def _load_request(path: Path) -> DecisionRequest:
    try:
        raw = json.loads(path.read_text())
        return DecisionRequest.model_validate(raw)
    except (json.JSONDecodeError, OSError, ValidationError) as exc:
        raise click.BadParameter(f"invalid request JSON in {path}: {exc}") from exc


def _load_policy(path: Path) -> PolicyBundle:
    try:
        raw = yaml.safe_load(path.read_text())
        return PolicyBundle.model_validate(raw)
    except (yaml.YAMLError, OSError, ValidationError) as exc:
        raise click.BadParameter(f"invalid policy bundle in {path}: {exc}") from exc


def _load_profile(profile_id: str) -> CostProfile:
    try:
        return load_cost_profile(profile_id)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _policy_decision(
    request: DecisionRequest, policy: PolicyBundle, profile: CostProfile
) -> DecisionResponse:
    """Delegate to the canonical PolicyEngine (never duplicate policy logic)."""
    engine = PolicyEngine(policy, profile)
    return engine.decide(request)


@click.command(name="simulate")
@click.option(
    "--trace",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL trace of decision events",
)
@click.option(
    "--policy",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="PolicyBundle YAML file",
)
@click.option("--cost-model", required=True, help="Cost profile id (see 'costs list')")
@_exit_on_error
def simulate(trace: Path, policy: Path, cost_model: str) -> None:
    """Replay a recorded workload under the best escalation threshold (P2.2).

    Prints {before, after, review_load_before, review_load_after} as JSON:
    before is the total cost at the policy's own threshold, after is the total
    cost at the best grid threshold. Delegates to the canonical ReplayLab;
    the CLI never re-implements replay math.
    """
    policy_model = _load_policy(policy)
    profile = _load_profile(cost_model)
    events: list[DecisionEvent] = []
    for line_no, record in enumerate(_read_trace(trace), start=1):
        event = _trace_to_event(record, line_no=line_no, path=trace)
        if event is not None:
            events.append(event)
    if not events:
        summary = {
            "before": 0.0,
            "after": 0.0,
            "review_load_before": 0.0,
            "review_load_after": 0.0,
        }
    else:
        probs = [e.calibrated_probability or 0.0 for e in events]
        errors = [(e.observed_outcome or {}).get("error", False) for e in events]
        severities = [_observed_severity(e) for e in events]
        best = fit_escalation_threshold(probs, errors, severities, profile)
        lab = ReplayLab(profile)
        outcome = lab.simulate(events, policy_model, best["best_threshold"])
        summary = {
            "before": outcome.before_loss,
            "after": outcome.after_loss,
            "review_load_before": outcome.before_review_load,
            "review_load_after": outcome.after_review_load,
        }
    click.echo(json.dumps(summary, sort_keys=True))


def _observed_severity(event: DecisionEvent) -> Severity:
    outcome = event.observed_outcome or {}
    raw = outcome.get("severity")
    try:
        return Severity(raw) if raw else Severity.LOW
    except ValueError:
        return Severity.LOW


def _trace_to_event(
    record: dict[str, Any], *, line_no: int, path: Path
) -> DecisionEvent | None:
    """Normalize a trace record into a DecisionEvent.

    Trace records may carry `severity` and `error` at the top level (legacy
    CLI format); these are folded into observed_outcome so the canonical
    ReplayLab severity/error resolution works. The calibrated risk is read
    from the top-level `calibrated_probability`, then the canonical
    `risk_features['calibrated_p']`. Returns None when the record has no
    calibrated probability (uncalibrated rows are skipped, not fatal).
    Raises click.BadParameter with the offending line number on records that
    fail schema validation.
    """
    risk_features = record.get("risk_features") or {}
    p = record.get("calibrated_probability")
    if p is None:
        p = risk_features.get("calibrated_p")
    if p is None:
        return None
    outcome = dict(record.get("observed_outcome") or {})
    if "error" not in outcome and record.get("error") is not None:
        outcome["error"] = record["error"]
    if "severity" not in outcome and record.get("severity") is not None:
        outcome["severity"] = record["severity"]
    normalized = {**record, "observed_outcome": outcome}
    try:
        return DecisionEvent.model_validate(normalized)
    except ValidationError as exc:
        raise click.BadParameter(
            f"invalid trace line {line_no} in {path}: {exc}"
        ) from exc


def _read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise click.BadParameter(f"invalid trace line {line_no} in {path}: {exc}") from exc
    return records


@click.command(name="version")
@_exit_on_error
def version() -> None:
    """Print the installed lossbench version."""
    click.echo(metadata.version("lossbench"))
