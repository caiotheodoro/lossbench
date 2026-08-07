"""Calibration methods: temperature scaling, Platt scaling, isotonic regression.

Fitting and prediction are deterministic for a fixed input. Outputs are
clipped to [0.001, 0.999]; inputs are validated against the length of the
correctness labels. All evaluation delegates to the P0 metrics in
``lossbench.metrics.calibration``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from lossbench.metrics.calibration import brier_score, ece

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None

_CLIP_MIN = 0.001
_CLIP_MAX = 0.999
_LOGIT_EPS = 1e-7
_LOG_PROB_EPS = 1e-12
_TEMP_BOUNDS = (0.05, 20.0)
_GRID_STEPS = 100


class CalibrationMethod(StrEnum):
    """Supported recalibration families."""

    TEMPERATURE = "temperature"
    PLATT = "platt"
    ISOTONIC = "isotonic"


def _as_arrays(
    confidences: Sequence[float], correct: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    confs = np.asarray(list(confidences), dtype=float)
    labels = np.asarray(list(correct), dtype=bool)
    if len(confs) != len(labels):
        raise ValueError("confidences and correct must have equal length")
    return confs, labels


def _clamp(value: float) -> float:
    return float(min(max(value, _CLIP_MIN), _CLIP_MAX))


def _temperature_nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(_sigmoid(logits / T), _LOG_PROB_EPS, 1.0 - _LOG_PROB_EPS)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(clipped / (1.0 - clipped))


def _fit_temperature(
    confidences: Sequence[float], correct: Sequence[bool]
) -> float:
    confs, labels = _as_arrays(confidences, correct)
    logits = _logit(confs)
    if minimize_scalar is not None:
        result = minimize_scalar(
            _temperature_nll,
            bounds=_TEMP_BOUNDS,
            args=(logits, labels),
            method="bounded",
        )
        return float(result.x)
    grid = np.linspace(_TEMP_BOUNDS[0], _TEMP_BOUNDS[1], _GRID_STEPS)
    losses = [_temperature_nll(t, logits, labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def _fit_platt(
    confidences: Sequence[float], correct: Sequence[bool]
) -> LogisticRegression:
    confs, labels = _as_arrays(confidences, correct)
    model = LogisticRegression(random_state=0, max_iter=1000)
    model.fit(confs.reshape(-1, 1), labels)
    return model


def _fit_isotonic(
    confidences: Sequence[float], correct: Sequence[bool]
) -> IsotonicRegression:
    confs, labels = _as_arrays(confidences, correct)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(confs, labels)
    return model


def fit_calibrator(
    method: CalibrationMethod,
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> Callable[[Sequence[float]], list[float]]:
    """Fit a calibration map on a calibration split.

    Returns a callable that maps raw confidences to calibrated probabilities
    clipped to [0.001, 0.999]. Temperature scaling minimizes the negative log
    likelihood over the temperature T in [0.05, 20.0] with
    scipy.optimize.minimize_scalar when scipy is available; otherwise it falls
    back to a 100-step grid search over the same range.
    """
    confs, labels = _as_arrays(confidences, correct)
    if len(confs) == 0:
        return lambda seq: list(seq)
    if method is CalibrationMethod.TEMPERATURE:
        temperature = _fit_temperature(confs, labels)

        def predict(seq: Sequence[float]) -> list[float]:
            raw = np.asarray(list(seq), dtype=float)
            return [_clamp(float(p)) for p in _sigmoid(_logit(raw) / temperature)]

        return predict
    if method is CalibrationMethod.PLATT:
        model = _fit_platt(confs, labels)

        def predict(seq: Sequence[float]) -> list[float]:
            raw = np.asarray(list(seq), dtype=float)
            probs = model.predict_proba(raw.reshape(-1, 1))[:, 1]
            return [_clamp(float(p)) for p in probs]

        return predict
    if method is CalibrationMethod.ISOTONIC:
        model = _fit_isotonic(confs, labels)

        def predict(seq: Sequence[float]) -> list[float]:
            raw = np.asarray(list(seq), dtype=float)
            return [_clamp(float(p)) for p in model.predict(raw)]

        return predict
    raise ValueError(f"unknown calibration method: {method}")


def recalibrate_on_window(
    method: CalibrationMethod,
    window_confidences: Sequence[float],
    window_correct: Sequence[bool],
    new_confidences: Sequence[float],
) -> list[float]:
    """Rolling recalibration: fit on the window, apply to new confidences."""
    predict = fit_calibrator(method, window_confidences, window_correct)
    return predict(new_confidences)


def calibrate_report(
    raw: Sequence[float],
    correct: Sequence[bool],
    calibrated: Sequence[float] | None = None,
    n_bins: int = 10,
) -> dict:
    """Calibration diagnostics for raw and optionally recalibrated scores.

    Returns ``{"raw_ece", "calibrated_ece", "brier_before", "brier_after",
    "n", "mean_shift"}`` where ``mean_shift`` is ``mean(calibrated) -
    mean(raw)`` when calibrated scores are given and 0.0 otherwise.
    """
    confs, labels = _as_arrays(raw, correct)
    report: dict = {
        "raw_ece": ece(list(confs), list(labels), n_bins)["ece"],
        "calibrated_ece": None,
        "brier_before": brier_score(list(confs), list(labels)),
        "brier_after": None,
        "n": len(confs),
        "mean_shift": 0.0,
    }
    if calibrated is not None:
        cal = np.asarray(list(calibrated), dtype=float)
        report["calibrated_ece"] = ece(list(cal), list(labels), n_bins)["ece"]
        report["brier_after"] = brier_score(list(cal), list(labels))
        if len(confs) > 0:
            report["mean_shift"] = float(np.mean(cal) - np.mean(confs))
    return report
