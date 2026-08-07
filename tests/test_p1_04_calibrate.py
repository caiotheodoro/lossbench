import numpy as np
import pytest

from lossbench.calibrate import (
    CalibrationMethod,
    calibrate_report,
    fit_calibrator,
    recalibrate_on_window,
)
from lossbench.metrics.calibration import ece


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _make_calibrated(rng, n=4000):
    p = rng.uniform(0.02, 0.98, n)
    correct = rng.random(n) < p
    return p, correct


def test_temperature_reduces_ece_on_known_distortion():
    rng = np.random.default_rng(10)
    p, correct = _make_calibrated(rng)
    distorted = np.clip(np.sqrt(p), 0.001, 0.999)
    predict = fit_calibrator(CalibrationMethod.TEMPERATURE, distorted, correct)
    calibrated = predict(distorted)
    raw_ece = ece(list(distorted), list(correct))["ece"]
    calibrated_ece = ece(calibrated, list(correct))["ece"]
    assert raw_ece > 0.1
    assert calibrated_ece < raw_ece


def test_platt_reduces_ece_on_logistic_distortion():
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 1.5, 4000)
    p = _sigmoid(z)
    correct = rng.random(4000) < p
    distorted = _sigmoid(z + 1.0)
    predict = fit_calibrator(CalibrationMethod.PLATT, distorted, correct)
    calibrated = predict(distorted)
    raw_ece = ece(list(distorted), list(correct))["ece"]
    calibrated_ece = ece(calibrated, list(correct))["ece"]
    assert calibrated_ece < raw_ece


def test_isotonic_reduces_ece():
    rng = np.random.default_rng(5)
    p, correct = _make_calibrated(rng)
    distorted = np.clip(p**1.5, 0.001, 0.999)
    predict = fit_calibrator(CalibrationMethod.ISOTONIC, distorted, correct)
    calibrated = predict(distorted)
    raw_ece = ece(list(distorted), list(correct))["ece"]
    calibrated_ece = ece(calibrated, list(correct))["ece"]
    assert calibrated_ece < raw_ece


def test_outputs_in_unit_range():
    rng = np.random.default_rng(11)
    p, correct = _make_calibrated(rng)
    inputs = [0.0, 0.001, 0.25, 0.5, 0.999, 1.0]
    for method in CalibrationMethod:
        predict = fit_calibrator(method, p, correct)
        assert all(0.0 <= v <= 1.0 for v in predict(inputs))


def test_deterministic():
    rng = np.random.default_rng(13)
    p, correct = _make_calibrated(rng)
    distorted = np.clip(np.sqrt(p), 0.001, 0.999)
    for method in CalibrationMethod:
        first = fit_calibrator(method, distorted, correct)(distorted)
        second = fit_calibrator(method, distorted, correct)(distorted)
        assert first == second


def test_recalibrate_on_window_reflects_shift():
    rng = np.random.default_rng(7)
    z_window = rng.normal(0.0, 1.2, 800)
    z_new = rng.normal(0.0, 1.2, 2000)
    window_correct = rng.random(800) < _sigmoid(z_window)
    new_correct = rng.random(2000) < _sigmoid(z_new)
    window_confidences = _sigmoid(z_window + 0.8)
    new_confidences = _sigmoid(z_new + 0.8)
    calibrated = recalibrate_on_window(
        CalibrationMethod.TEMPERATURE,
        window_confidences,
        window_correct,
        new_confidences,
    )
    raw_ece = ece(list(new_confidences), list(new_correct))["ece"]
    calibrated_ece = ece(calibrated, list(new_correct))["ece"]
    assert calibrated_ece < raw_ece


def test_calibrate_report_shape():
    rng = np.random.default_rng(17)
    p, correct = _make_calibrated(rng, n=2000)
    distorted = np.clip(np.sqrt(p), 0.001, 0.999)
    predict = fit_calibrator(CalibrationMethod.ISOTONIC, distorted, correct)
    calibrated = predict(distorted)
    report = calibrate_report(distorted, correct, calibrated, n_bins=10)
    assert set(report) == {
        "raw_ece",
        "calibrated_ece",
        "brier_before",
        "brier_after",
        "n",
        "mean_shift",
    }
    assert report["n"] == 2000
    assert 0.0 <= report["raw_ece"] <= 1.0
    assert 0.0 <= report["calibrated_ece"] <= 1.0
    assert report["brier_after"] < report["brier_before"]
    assert report["mean_shift"] == pytest.approx(
        float(np.mean(calibrated) - np.mean(distorted))
    )
    raw_only = calibrate_report(distorted, correct)
    assert raw_only["calibrated_ece"] is None
    assert raw_only["brier_after"] is None
    assert raw_only["mean_shift"] == 0.0


def test_empty_inputs_safe():
    for method in CalibrationMethod:
        identity = fit_calibrator(method, [], [])
        assert identity([0.5, 0.9]) == [0.5, 0.9]
    report = calibrate_report([], [])
    assert report["raw_ece"] == 0.0
    assert report["brier_before"] == 0.0
    assert report["calibrated_ece"] is None
    assert report["brier_after"] is None
    assert report["n"] == 0
    assert report["mean_shift"] == 0.0
