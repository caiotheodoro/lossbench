"""Calibration metrics: ECE, reliability curves, Brier score.

Pure functions over (confidence, correctness) pairs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _bin_index(conf: float, n_bins: int) -> int:
    idx = int(conf * n_bins)
    return min(idx, n_bins - 1)


def reliability_curve(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[dict]:
    """Per-bin {conf_min, conf_max, conf_mean, accuracy, count}."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        return []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct, strict=True):
        bins[_bin_index(conf, n_bins)].append((conf, bool(ok)))
    curve: list[dict] = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        conf_mean = float(np.mean([c for c, _ in bucket]))
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        curve.append(
            {
                "bin": i,
                "conf_min": round(i / n_bins, 3),
                "conf_max": round((i + 1) / n_bins, 3),
                "conf_mean": round(conf_mean, 4),
                "accuracy": round(acc, 4),
                "count": len(bucket),
            }
        )
    return curve


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> dict:
    """Expected calibration error plus the reliability curve.

    ECE = sum_b (n_b/N) * |acc_b - conf_b|.
    """
    curve = reliability_curve(confidences, correct, n_bins)
    n = len(confidences)
    total = 0.0
    for point in curve:
        total += (point["count"] / n) * abs(point["accuracy"] - point["conf_mean"])
    return {"ece": round(total, 4), "n_bins": n_bins, "curve": curve}


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of confidence vs outcome indicator."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        return 0.0
    errs = np.mean(
        [(c - int(ok)) ** 2 for c, ok in zip(confidences, correct, strict=True)]
    )
    return float(errs)
