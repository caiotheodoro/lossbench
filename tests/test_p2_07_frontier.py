"""P2.7 frontier report wiring tests."""

from __future__ import annotations

from lossbench.costs.registry import load_cost_profile
from lossbench.metrics.sensitivity import cost_sensitivity_curves
from lossbench.report.frontier import frontier_report
from lossbench.schema import Severity

N = 600
SEVS = [Severity.LOW] * (N // 2) + [Severity.HIGH] * (N // 2)

PATTERNS = {
    "model_a": {
        "errors": [i < 30 or (540 <= i < 560) for i in range(N)],
        "severities_mix": {"LOW": 0.9, "HIGH": 0.1},
    },
    "model_b": {
        "errors": [i < 200 or (540 <= i < 550) for i in range(N)],
        "severities_mix": {"LOW": 0.9, "HIGH": 0.1},
    },
}


def test_frontier_report_returns_dict_and_markdown():
    report, markdown = frontier_report(
        model_losses={"model_a": 10.5, "model_b": 20.0},
        severities=SEVS,
        model_error_patterns=PATTERNS,
    )
    assert report["losses"]["model_a"] == 10.5
    assert "| Model | Loss |" in markdown
    assert "model_a" in markdown
    assert "sensitivities" in report


def test_markdown_contains_sensitivity_tables():
    _, markdown = frontier_report(
        model_losses={"model_a": 1.0}, severities=SEVS, model_error_patterns=PATTERNS
    )
    assert "| ratio | loss |" in markdown


def test_optional_sections_omitted():
    report, markdown = frontier_report(model_losses={"m": 1.0}, severities=SEVS)
    assert "sensitivities" not in report
    assert "calibration" not in report
    assert "| ratio | loss |" not in markdown


def test_metadata_embedded():
    report, markdown = frontier_report(
        model_losses={"m": 1.0}, severities=SEVS, suite="finance-v1", cost_model="flat"
    )
    assert report["metadata"]["suite"] == "finance-v1"
    assert report["metadata"]["cost_model"] == "flat"


def test_honest_limits_rendered():
    _, markdown = frontier_report(
        model_losses={"m": 1.0},
        severities=SEVS,
        honest_limits=["rankings depend on the cost regime"],
    )
    assert "rankings depend on the cost regime" in markdown


def test_losses_sorted_ascending():
    report, _ = frontier_report(
        model_losses={"b": 5.0, "a": 1.0, "c": 3.0}, severities=SEVS
    )
    assert list(report["losses"]) == ["a", "c", "b"]


def test_sensitivities_match_core_module():
    report, _ = frontier_report(
        model_losses={"model_a": 1.0},
        severities=SEVS,
        model_error_patterns=PATTERNS,
        cost_ratios=(1.0, 10.0),
    )
    core = cost_sensitivity_curves(
        PATTERNS, SEVS, (1.0, 10.0), base_profile=load_cost_profile("reconciliation")
    )
    assert report["sensitivities"]["model_a"] == core["model_a"]


def test_deterministic_markdown():
    report, _ = frontier_report(model_losses={"a": 1.0}, severities=SEVS)
    from lossbench.report.generator import render_markdown

    assert render_markdown(report) == render_markdown(report)
