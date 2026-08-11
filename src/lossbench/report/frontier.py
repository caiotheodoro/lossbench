"""P2.7 cost-sensitivity frontier + report wiring.

Assembles the canonical benchmark report: per-model severity-weighted loss,
cost-sensitivity curves, and (when available) calibration/deferral data —
the artifact that shows WHERE rankings flip as cost ratios change.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lossbench.metrics.sensitivity import cost_sensitivity_curves
from lossbench.report.generator import build_report, render_markdown
from lossbench.schema import Severity


def frontier_report(
    model_losses: dict[str, float],
    severities: Sequence[Severity],
    model_error_patterns: dict[str, dict] | None = None,
    cost_ratios: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 100.0),
    ece_results: dict[str, dict] | None = None,
    deferral_results: dict[str, float] | None = None,
    honest_limits: list[str] | None = None,
    suite: str = "finance-v1",
    cost_model: str = "reconciliation",
) -> tuple[dict, str]:
    """Build (report_dict, markdown) for the benchmark release.

    model_error_patterns feeds cost_sensitivity_curves (see P1.13); when
    given, the report gains a sensitivities section and the honest-limits
    note that rankings depend on the cost regime. model_losses must contain
    at least one model.
    """
    report = build_report(
        model_losses=model_losses,
        sensitivities=(
            cost_sensitivity_curves(model_error_patterns, severities, cost_ratios)
            if model_error_patterns
            else None
        ),
        ece_results=ece_results,
        deferral_results=deferral_results,
        metadata={
            "generated_at": datetime.now(UTC).isoformat(),
            "suite": suite,
            "cost_model": cost_model,
        },
        honest_limits=honest_limits or [],
    )
    return report, render_markdown(report)
