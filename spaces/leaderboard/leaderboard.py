"""LossBench P3.1 leaderboard: pure logic for the Hugging Face Gradio Space.

The Space consumes a leaderboard JSON — a static artifact, no live database —
and renders a severity-weighted leaderboard with cost-sensitivity crossovers
and honest limits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lossbench.report.templates import md_table


@dataclass(frozen=True)
class LeaderboardRow:
    """One model's row on the severity-weighted leaderboard."""

    model_id: str
    severity_weighted_loss: float
    pass_k: float | None = None
    ece: float | None = None
    escalated: float | None = None
    total_cost: float | None = None


def _opt_number(value: Any, what: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be numeric, got {value!r}") from exc


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def load_leaderboard(path: str | Path) -> list[LeaderboardRow]:
    """Parse a leaderboard JSON: {"models": [...]} sorted by loss ascending.

    Each model entry carries model_id and the severity-weighted loss (both
    required) plus the optional pass_k, ece, escalated, and total_cost fields;
    absent optional fields become None. The loss key is ``severity_weighted_loss``
    — the one name the run artifacts and ``scripts/full_run.py`` emit — with the
    legacy ``loss`` alias still accepted. Raises ValueError on a missing
    model_id or loss.
    """
    with Path(path).open(encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        raise ValueError("leaderboard JSON must contain a 'models' list")
    rows: list[LeaderboardRow] = []
    for entry in doc["models"]:
        if not isinstance(entry, dict):
            raise ValueError("every leaderboard entry must be an object")
        model_id = entry.get("model_id")
        loss = entry.get("severity_weighted_loss")
        if loss is None:
            loss = entry.get("loss")
        if model_id is None:
            raise ValueError("missing model_id")
        if loss is None:
            raise ValueError(
                f"missing severity_weighted_loss for model '{model_id}'"
            )
        rows.append(
            LeaderboardRow(
                model_id=str(model_id),
                severity_weighted_loss=_opt_number(loss, "severity_weighted_loss"),
                pass_k=_opt_number(entry.get("pass_k"), "pass_k"),
                ece=_opt_number(entry.get("ece"), "ece"),
                escalated=_opt_number(entry.get("escalated"), "escalated"),
                total_cost=_opt_number(entry.get("total_cost"), "total_cost"),
            )
        )
    return sorted(rows, key=lambda row: (row.severity_weighted_loss, row.model_id))


def render_table(rows: Sequence[LeaderboardRow]) -> str:
    """Render rows as a markdown table; '-' for missing optional fields.

    Columns are Model, Loss, pass^k, ECE, Escalated, Cost; loss and other
    numerics are formatted to 4 decimals, counts to integers.
    """
    header = ("Model", "Loss", "pass^k", "ECE", "Escalated", "Cost")
    body = tuple(
        (
            row.model_id,
            f"{row.severity_weighted_loss:.4f}",
            _fmt(row.pass_k),
            _fmt(row.ece),
            f"{row.escalated:.0f}" if row.escalated is not None else "-",
            _fmt(row.total_cost),
        )
        for row in rows
    )
    return md_table(header, body)


def crossover_summary(sensitivities: dict[str, list[dict[str, float]]] | None) -> str:
    """Summarize cost-sensitivity ranking flips from a frontier report.

    Crossing detection mirrors ranking_stability: for each pair of models, the
    first ratio at which their loss ranking swaps relative to the first ratio.
    Emits one "rankings flip at ratio X between A and B" line per pair, or
    "no crossovers" when none exist; "no sensitivities" when input is None.
    """
    if sensitivities is None:
        return "no sensitivities"
    loss_at: dict[str, dict[float, float]] = {}
    for model_id, points in sensitivities.items():
        loss_at[model_id] = {point["ratio"]: point["loss"] for point in points}
    if not loss_at:
        return "no crossovers"
    common_ratios = sorted(set.intersection(*(set(points) for points in loss_at.values())))
    if not common_ratios:
        return "no crossovers"
    rankings = {
        ratio: sorted(loss_at, key=lambda model_id: (loss_at[model_id][ratio], model_id))
        for ratio in common_ratios
    }
    lines: list[str] = []
    model_ids = sorted(loss_at)
    for i, a in enumerate(model_ids):
        for b in model_ids[i + 1 :]:
            order_at_first = (
                rankings[common_ratios[0]].index(a) < rankings[common_ratios[0]].index(b)
            )
            for ratio in common_ratios:
                order_now = rankings[ratio].index(a) < rankings[ratio].index(b)
                if order_now != order_at_first:
                    lines.append(f"rankings flip at ratio {ratio:g} between {a} and {b}")
                    break
    return "\n".join(lines) if lines else "no crossovers"


def load_banner(path: str | Path) -> str | None:
    """Return the leaderboard JSON's ``banner`` string, or None if absent.

    A banner marks data that is not a real benchmark result — a synthetic demo
    fixture or stub pipeline smoke output. The UI renders it as a loud, visible
    warning; it is never a silent fallback.
    """
    with Path(path).open(encoding="utf-8") as fh:
        doc = json.load(fh)
    banner = doc.get("banner")
    return str(banner) if banner else None


def _load_extras(path: str | Path) -> tuple[dict[str, list[dict[str, float]]] | None, list[str]]:
    """Read optional sensitivities and honest_limits from a leaderboard JSON."""
    with Path(path).open(encoding="utf-8") as fh:
        doc = json.load(fh)
    sensitivities = doc.get("sensitivities")
    honest_limits = doc.get("honest_limits")
    if not isinstance(honest_limits, list):
        honest_limits = []
    return sensitivities, list(honest_limits)


def _format_honest_limits(limits: Sequence[str]) -> str:
    """Render honest limits as a markdown bullet list, or a placeholder."""
    if not limits:
        return "_No honest limits recorded for this run._"
    return "## Honest Limits\n\n" + "\n".join(f"- {item}" for item in limits)
