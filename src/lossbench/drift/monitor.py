"""Loss-distribution drift monitoring (design C4).

PSI on the expected-loss distribution triggers fail-safe escalation; KS on
calibrated probabilities and realized-ECE deltas trigger recalibration.
Pure numpy/scipy, deterministic, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from lossbench.metrics.calibration import ece
from lossbench.schema import DecisionEvent

_EPSILON = 1e-6


@dataclass(frozen=True)
class DriftReport:
    """One drift verdict for a monitored feature."""

    feature: str
    statistic: float
    p_value: float
    alert: bool
    direction: str


def psi(expected: Sequence[float], actual: Sequence[float], n_bins: int = 10) -> float:
    """Population Stability Index over `n_bins` bins with epsilon-smoothed
    expected proportions.

    Shared bin edges cover the combined range of both samples. Expected-empty
    bins get epsilon = 1e-6. psi = sum((a - e) * ln(a / e)). Identical samples
    yield 0.0; empty inputs yield 0.0.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.size == 0 or a.size == 0:
        return 0.0
    lo = float(min(e.min(), a.min()))
    hi = float(max(e.max(), a.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0
    pad = max(1e-9, (hi - lo) * 1e-9)
    edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
    e_hist, _ = np.histogram(e, bins=edges)
    a_hist, _ = np.histogram(a, bins=edges)
    e_prop = np.maximum(e_hist / e.size, _EPSILON)
    a_prop = np.maximum(a_hist / a.size, _EPSILON)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def ks_drift(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov drift; returns (statistic, p_value).

    Identical samples give statistic ~ 0.0 and p_value == 1.0.
    """
    result = stats.ks_2samp(a, b)
    return float(result.statistic), float(result.pvalue)


def realized_ece(events: Sequence[DecisionEvent], n_bins: int = 10) -> float:
    """ECE over events whose observed_outcome carries an 'error' key.

    Confidence is calibrated_probability (0.5 when None); the positive class
    is error=True. Events without an 'error' key are ignored; no usable
    events yields 0.0.
    """
    probs: list[float] = []
    correct: list[bool] = []
    for event in events:
        outcome = event.observed_outcome or {}
        if "error" in outcome:
            p = event.calibrated_probability if event.calibrated_probability is not None else 0.5
            probs.append(p)
            correct.append(bool(outcome["error"]))
    if not probs:
        return 0.0
    return float(ece(probs, correct, n_bins)["ece"])


class DriftMonitor:
    """Compares a window of DecisionEvents against a fitted baseline."""

    def __init__(
        self,
        psi_threshold: float = 0.25,
        ks_p_threshold: float = 0.01,
        ece_delta_threshold: float = 0.10,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_p_threshold = ks_p_threshold
        self.ece_delta_threshold = ece_delta_threshold
        self._baseline_expected_loss: np.ndarray | None = None
        self._baseline_calibrated_p: np.ndarray | None = None
        self._baseline_ece: float | None = None

    def fit_baseline(self, events: Sequence[DecisionEvent]) -> None:
        """Record the reference distributions and baseline realized ECE.

        expected_loss defaults to 0.0 and calibrated_probability to 0.5 when
        None. Baseline ECE uses only events with an observed_outcome 'error'
        key; when none exist the realized-ECE path stays inactive.
        """
        self._baseline_expected_loss = np.asarray(
            [event.expected_loss if event.expected_loss is not None else 0.0 for event in events],
            dtype=float,
        )
        self._baseline_calibrated_p = np.asarray(
            [
                event.calibrated_probability
                if event.calibrated_probability is not None
                else 0.5
                for event in events
            ],
            dtype=float,
        )
        has_error_labels = any(
            event.observed_outcome and "error" in event.observed_outcome for event in events
        )
        self._baseline_ece = realized_ece(events) if has_error_labels else None

    def detect(self, events: Sequence[DecisionEvent]) -> list[DriftReport]:
        """Per-feature drift verdicts versus the fitted baseline.

        No fitted baseline or an empty window yields []. Features:
        expected_loss_distribution (PSI, 'fail_safe_escalate' when psi >
        psi_threshold), calibrated_p (KS, 'recalibrate' when p <
        ks_p_threshold), realized_ece (delta vs baseline ECE, 'recalibrate'
        when delta > ece_delta_threshold).
        """
        if (
            self._baseline_expected_loss is None
            or self._baseline_calibrated_p is None
            or self._baseline_expected_loss.size == 0
        ):
            return []
        if not events:
            return []

        window_loss = np.asarray(
            [event.expected_loss if event.expected_loss is not None else 0.0 for event in events],
            dtype=float,
        )
        window_p = np.asarray(
            [
                event.calibrated_probability
                if event.calibrated_probability is not None
                else 0.5
                for event in events
            ],
            dtype=float,
        )

        reports: list[DriftReport] = []

        psi_stat = psi(self._baseline_expected_loss, window_loss)
        loss_alert = psi_stat > self.psi_threshold
        reports.append(
            DriftReport(
                feature="expected_loss_distribution",
                statistic=float(psi_stat),
                p_value=-1.0,
                alert=loss_alert,
                direction="fail_safe_escalate" if loss_alert else "ok",
            )
        )

        ks_stat, ks_p = ks_drift(self._baseline_calibrated_p, window_p)
        p_alert = ks_p < self.ks_p_threshold
        reports.append(
            DriftReport(
                feature="calibrated_p",
                statistic=ks_stat,
                p_value=ks_p,
                alert=p_alert,
                direction="recalibrate" if p_alert else "ok",
            )
        )

        if self._baseline_ece is not None and any(
            event.observed_outcome and "error" in event.observed_outcome for event in events
        ):
            ece_delta = realized_ece(events) - self._baseline_ece
            ece_alert = ece_delta > self.ece_delta_threshold
            reports.append(
                DriftReport(
                    feature="realized_ece",
                    statistic=float(ece_delta),
                    p_value=-1.0,
                    alert=ece_alert,
                    direction="recalibrate" if ece_alert else "ok",
                )
            )

        return reports

    def escalation_override(self, reports: Sequence[DriftReport]) -> bool:
        """True when any report demands fail-safe escalation."""
        return any(report.direction == "fail_safe_escalate" for report in reports)

    def recalibration_needed(self, reports: Sequence[DriftReport]) -> bool:
        """True when any report demands recalibration."""
        return any(report.direction == "recalibrate" for report in reports)
