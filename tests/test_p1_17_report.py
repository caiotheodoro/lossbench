"""P1.17 report generator tests."""

from __future__ import annotations

import pytest

from lossbench.report import build_report, render_html, render_markdown


def test_markdown_contains_loss_table():
    md = render_markdown(build_report({"a": 1.5, "b": 2.5}))
    assert "| Model | Loss |" in md
    assert md.index("| a |") < md.index("| b |")


def test_markdown_sections_present_when_data_given():
    report = build_report(
        {"a": 1.0, "b": 2.0},
        sensitivities={"a": [{"ratio": 1.0, "loss": 3.0}]},
        ece_results={"a": {"ece": 0.05, "n": 100}},
        deferral_results={
            "escalation_precision": 0.8,
            "escalation_recall": 0.7,
            "missed_high_loss_rate": 0.1,
        },
        honest_limits=["no limit claims"],
    )
    md = render_markdown(report)
    assert "## Losses" in md
    assert "Cost Sensitivity" in md
    assert "## Calibration" in md
    assert "## Deferral" in md
    assert "## Honest Limits" in md
    bare = render_markdown(build_report({"a": 1.0}))
    assert "Cost Sensitivity" not in bare
    assert "## Calibration" not in bare
    assert "## Deferral" not in bare
    assert "## Honest Limits" not in bare


def test_html_valid_structure():
    html = render_html(build_report({"a": 1.5, "b": 2.5}))
    assert "<html>" in html
    assert "<table>" in html
    assert "</table>" in html
    assert "1.5000" in html
    assert "2.5000" in html


def test_deterministic():
    report = build_report(
        {"a": 1.5, "b": 0.5},
        sensitivities={"a": [{"ratio": 1.0, "loss": 3.0}]},
        ece_results={"a": {"ece": 0.05, "n": 10}},
        deferral_results={
            "escalation_precision": 0.8,
            "escalation_recall": 0.7,
            "missed_high_loss_rate": 0.1,
        },
    )
    assert render_markdown(report) == render_markdown(report)
    assert render_html(report) == render_html(report)


def test_honest_limits_rendered():
    md = render_markdown(build_report({"a": 1.0}, honest_limits=["x"]))
    assert "1. x" in md


def test_invalid_losses_raise():
    with pytest.raises(ValueError):
        build_report({"a": "nan"})


def test_empty_report_ok():
    report = build_report({})
    assert report["losses"] == {}
    assert report["honest_limits"] == []
    md = render_markdown(report)
    assert "| Model | Loss |" in md
    body = md.split("| --- | --- |", 1)[1]
    assert not any(line.startswith("|") for line in body.splitlines())


def test_sensitivity_table_rows():
    sensitivities = {
        "a": [{"ratio": 1.0, "loss": 5.0}, {"ratio": 2.0, "loss": 9.0}],
        "b": [
            {"ratio": 1.0, "loss": 6.0},
            {"ratio": 2.0, "loss": 8.0},
            {"ratio": 5.0, "loss": 10.0},
        ],
    }
    md = render_markdown(build_report({"a": 1.0, "b": 2.0}, sensitivities=sensitivities))
    for model in ("a", "b"):
        block = md.split(f"## Cost Sensitivity: {model}", 1)[1]
        next_heading = block.find("## ")
        block = block if next_heading == -1 else block[:next_heading]
        assert "| ratio | loss |" in block
        pipe_lines = [line for line in block.splitlines() if line.startswith("|")]
        assert len(pipe_lines) - 2 == len(sensitivities[model])
