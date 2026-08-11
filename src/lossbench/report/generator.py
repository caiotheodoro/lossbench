"""Canonical report assembly and markdown/HTML rendering.

build_report produces the canonical REPORT SCHEMA dict; render_markdown and
render_html turn any such dict into deterministic, self-contained output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from typing import Any

from lossbench.report import templates as _templates

DEFAULT_TITLE = "LossBench Report"


def _check_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be numeric")
    if not isfinite(float(value)):
        raise ValueError(f"{what} must be finite")
    return float(value)


def _checked_losses(model_losses: dict[str, Any]) -> dict[str, float]:
    for model_id, loss in model_losses.items():
        _check_number(loss, f"loss for model '{model_id}'")
    return dict(sorted(model_losses.items(), key=lambda kv: (float(kv[1]), kv[0])))


def _checked_sensitivities(sensitivities: dict[str, Any]) -> dict[str, Any]:
    for model_id, rows in sensitivities.items():
        if not isinstance(rows, list):
            raise ValueError(f"sensitivity rows for model '{model_id}' must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"sensitivity row for model '{model_id}' must be a dict")
            _check_number(row.get("ratio"), f"ratio for model '{model_id}'")
            _check_number(row.get("loss"), f"loss for model '{model_id}'")
    return sensitivities


def _checked_calibration(ece_results: dict[str, Any]) -> dict[str, Any]:
    for model_id, entry in ece_results.items():
        if not isinstance(entry, dict):
            raise ValueError(f"calibration entry for model '{model_id}' must be a dict")
        _check_number(entry.get("ece"), f"ece for model '{model_id}'")
        _check_number(entry.get("n"), f"n for model '{model_id}'")
    return ece_results


def _checked_deferral(deferral_results: dict[str, Any]) -> dict[str, Any]:
    for key in ("escalation_precision", "escalation_recall", "missed_high_loss_rate"):
        _check_number(deferral_results.get(key), f"deferral {key}")
    return deferral_results


def build_report(
    model_losses: dict[str, float],
    sensitivities: dict[str, list[dict[str, float]]] | None = None,
    ece_results: dict[str, dict] | None = None,
    deferral_results: dict[str, float] | None = None,
    metadata: dict[str, str] | None = None,
    honest_limits: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical report dict from metric results."""
    metadata = dict(metadata or {})
    metadata.setdefault("generated_at", datetime.now(UTC).isoformat(timespec="seconds"))
    metadata.setdefault("suite", "")
    metadata.setdefault("cost_model", "")
    report: dict[str, Any] = {
        "title": metadata.get("title") or DEFAULT_TITLE,
        "metadata": metadata,
        "losses": _checked_losses(model_losses),
        "honest_limits": list(honest_limits or []),
    }
    if sensitivities is not None:
        report["sensitivities"] = _checked_sensitivities(sensitivities)
    if ece_results is not None:
        report["calibration"] = _checked_calibration(ece_results)
    if deferral_results is not None:
        report["deferral"] = _checked_deferral(deferral_results)
    return report


def _sections(report: dict[str, Any]) -> list[_templates._templates.Section]:
    sections: list[_templates.Section] = []
    metadata = report.get("metadata") or {}
    if metadata:
        rows = tuple(
            (str(key), str(value))
            for key, value in sorted(metadata.items())
            if key != "title"
        )
        sections.append(_templates.Section("metadata", "Metadata", "bullets", rows=rows))
    if "losses" in report:
        rows = tuple(
            (str(model_id), _templates.format_number(loss))
            for model_id, loss in report["losses"].items()
        )
        sections.append(_templates.Section("losses", "Losses", "table", ("Model", "Loss"), rows))
    sensitivities = report.get("sensitivities")
    if sensitivities:
        for model_id in sorted(sensitivities):
            rows = tuple(
                (_templates.format_number(point["ratio"]), _templates.format_number(point["loss"]))
                for point in sensitivities[model_id]
            )
            sections.append(
                _templates.Section(
                    f"sensitivity-{model_id}",
                    f"Cost Sensitivity: {model_id}",
                    "table",
                    ("ratio", "loss"),
                    rows,
                )
            )
    calibration = report.get("calibration")
    if calibration:
        rows = tuple(
            (str(model_id), _templates.format_number(entry["ece"]), str(entry["n"]))
            for model_id, entry in sorted(calibration.items())
        )
        sections.append(
            _templates.Section("calibration", "Calibration", "table", ("Model", "ECE", "n"), rows)
        )
    deferral = report.get("deferral")
    if deferral:
        rows = (
            ("escalation_precision", _templates.format_number(deferral["escalation_precision"])),
            ("escalation_recall", _templates.format_number(deferral["escalation_recall"])),
            ("missed_high_loss_rate", _templates.format_number(deferral["missed_high_loss_rate"])),
        )
        sections.append(_templates.Section("deferral", "Deferral", "bullets", rows=rows))
    limits = report.get("honest_limits")
    if limits:
        rows = tuple((str(item),) for item in limits)
        sections.append(_templates.Section("honest-limits", "Honest Limits", "numbered", rows=rows))
    return sections


def render_markdown(report: dict[str, Any]) -> str:
    """Deterministic markdown from a report dict (see REPORT SCHEMA).

    _templates.Sections in fixed order; numbers rendered with 4 decimals; tables with
    aligned pipes.
    """
    return _templates.render_markdown(str(report.get("title") or DEFAULT_TITLE), _sections(report))


def render_html(report: dict[str, Any]) -> str:
    """Minimal self-contained HTML (inline CSS) with the same content."""
    return _templates.render_html(str(report.get("title") or DEFAULT_TITLE), _sections(report))
