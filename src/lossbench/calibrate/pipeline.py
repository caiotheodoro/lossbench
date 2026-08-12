"""Production calibration loop: fit calibrator and escalation threshold from labels.

The pipeline is the operational loop: ledger labels (outcomes) arrive, the
calibrator is refit on labeled events only, all events get calibrated
probabilities (unlabeled keep their raw confidence), and a new escalation
threshold is fitted from the labeled subset. Everything is deterministic:
no shuffling, and every fitted component (scipy minimize_scalar, sklearn
models seeded with random_state=0) is deterministic for fixed inputs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from lossbench.calibrate.methods import (
    CalibrationMethod,
    calibrate_report,
    fit_calibrator,
)
from lossbench.features.extract import extract_risk_features
from lossbench.ledger.store import AuditLedger
from lossbench.policy.fit import fit_escalation_threshold
from lossbench.schema import CostProfile, DecisionEvent, PolicyBundle, Severity


@dataclass
class PipelineResult:
    """Output of one calibration-pipeline run over a batch of decision events."""

    method: CalibrationMethod
    calibrated: list[float]
    threshold: float
    report: dict
    n_labeled: int
    n_unlabeled: int


def _label_of(
    event: DecisionEvent, label_fn: Callable[[DecisionEvent], bool] | None
) -> tuple[bool, bool]:
    if label_fn is not None:
        value = bool(label_fn(event))
        return value, value
    outcome = event.observed_outcome or {}
    return "error" in outcome, outcome.get("error") is True


def _identity(confidences: Sequence[float]) -> list[float]:
    """Identity calibrator for degenerate (n<2 labeled) inputs."""
    return list(confidences)


def _resolve_severity(event: DecisionEvent) -> Severity:
    raw = (event.observed_outcome or {}).get("severity")
    if raw is None:
        return Severity.LOW
    if isinstance(raw, Severity):
        return raw
    try:
        return Severity(str(raw))
    except ValueError:
        return Severity.LOW


def run_calibration_pipeline(
    events: Sequence[DecisionEvent],
    cost_profile: CostProfile,
    method: CalibrationMethod = CalibrationMethod.TEMPERATURE,
    label_fn: Callable[[DecisionEvent], bool] | None = None,
) -> PipelineResult:
    """Fit a calibrator and escalation threshold from labeled events.

    A decision is labeled when ``label_fn`` is given and returns True, or when
    ``observed_outcome`` contains the ``error`` key; unlabeled events never
    enter a fit, and events with no calibrated probability are excluded from
    the fit as well (a missing p is unknown risk, not p=0). The raw
    confidence signal is the ``confidence`` feature from
    ``extract_risk_features``. The calibrator is fit on a random half of the
    labeled subset and evaluated on BOTH the fit half and the held-out half:
    the report's ``calibrated_ece`` is the HELD-OUT ECE (never in-sample —
    isotonic collapses to 0 in-sample while leaving real error held-out), and
    ``calibrated_ece_fit`` is reported alongside for transparency. Unlabeled
    events keep their raw confidence. The escalation threshold is
    grid-searched on the labeled subset only, with severities resolved per
    event from ``observed_outcome["severity"]``, defaulting to LOW. The
    report covers the labeled subset only.
    """
    confidences = [extract_risk_features(event)["confidence"] for event in events]
    marks = [_label_of(event, label_fn) for event in events]
    is_labeled = [marked for marked, _ in marks]
    correct = [label for _, label in marks]
    has_p = [c is not None for c in confidences]
    fit_mask = [
        marked and p for marked, p in zip(is_labeled, has_p, strict=True)
    ]
    labeled_confidences = [
        conf for conf, keep in zip(confidences, fit_mask, strict=True) if keep
    ]
    labeled_correct = [
        label for label, keep in zip(correct, fit_mask, strict=True) if keep
    ]
    n_labeled = sum(fit_mask)
    if n_labeled >= 2:
        rng = random.Random(0)
        order = list(range(len(labeled_confidences)))
        rng.shuffle(order)
        split = max(1, len(order) // 2)
        fit_idx = order[:split]
        held_idx = order[split:]
        fit_conf = [labeled_confidences[i] for i in fit_idx]
        fit_correct = [labeled_correct[i] for i in fit_idx]
        held_conf = [labeled_confidences[i] for i in held_idx]
        held_correct = [labeled_correct[i] for i in held_idx]
        predict = fit_calibrator(method, fit_conf, fit_correct)
        calibrated_fit = predict(fit_conf)
        calibrated_held = predict(held_conf)
        fit_ece = calibrate_report(fit_conf, fit_correct, calibrated_fit)["calibrated_ece"]
        held_report = calibrate_report(held_conf, held_correct, calibrated_held)
        report = held_report
        report["calibrated_ece_fit"] = fit_ece
        calibrated_labeled = list(calibrated_fit) + list(calibrated_held)
    else:
        predict = _identity
        report = calibrate_report(
            labeled_confidences, labeled_correct, list(labeled_confidences)
        )
        report["calibrated_ece_fit"] = report["calibrated_ece"]
        calibrated_labeled = list(labeled_confidences)
    calibrated = [
        predict([conf])[0] if keep else conf
        for conf, keep in zip(confidences, fit_mask, strict=True)
    ]
    threshold = 0.0
    if n_labeled:
        severities = [
            _resolve_severity(event)
            for event, keep in zip(events, fit_mask, strict=True)
            if keep
        ]
        threshold = float(
            fit_escalation_threshold(
                calibrated_labeled, labeled_correct, severities, cost_profile
            )["best_threshold"]
        )
    return PipelineResult(
        method=method,
        calibrated=calibrated,
        threshold=threshold,
        report=report,
        n_labeled=n_labeled,
        n_unlabeled=len(events) - n_labeled,
    )


def fit_policy_from_ledger(
    ledger: AuditLedger,
    cost_profile: CostProfile,
    policy_id: str,
    escalation_threshold: float | None = None,
    limit: int = 1000,
) -> PolicyBundle:
    """Refit a policy bundle from labeled events stored in a ledger.

    Pulls up to ``limit`` events in append order, runs the temperature
    calibration pipeline over them, and assembles a PolicyBundle. The bundle
    uses the provided escalation threshold when given, otherwise the fitted
    one (0.0 when no labeled events are available). Revision is
    "fitted-YYYYMMDDHHMMSS".
    """
    rows = ledger._conn.execute(
        "SELECT event_json FROM events ORDER BY seq LIMIT ?", [limit]
    ).fetchall()
    events = [DecisionEvent.model_validate_json(row[0]) for row in rows]
    result = run_calibration_pipeline(events, cost_profile, CalibrationMethod.TEMPERATURE)
    fitted_threshold = escalation_threshold
    if fitted_threshold is None:
        fitted_threshold = result.threshold if result.n_labeled else 0.0
    return PolicyBundle(
        id=policy_id,
        revision=datetime.now().strftime("fitted-%Y%m%d%H%M%S"),
        cost_model_id=cost_profile.id,
        escalation_threshold=float(fitted_threshold),
    )
