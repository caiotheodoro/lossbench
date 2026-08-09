"""Trajectory Proper Score (TPS): proper-scoring evaluation of trajectories.

Pure functions over DecisionEvent sequences; no I/O. The TPS scores a
forecaster on its per-step predicted probability of trajectory success,
making it a strictly proper objective for the prefix-conditioned success
process.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lossbench.schema import DecisionEvent, DecisionKind

Q_CLAMP_MIN = 0.01
Q_CLAMP_MAX = 0.99
Q_UNCERTAINTY = 0.5
ECE_N_BINS = 10


def trajectory_success_probs(events: Sequence[DecisionEvent]) -> list[float]:
    """Prefix-conditioned success probabilities q_t = P(success | history_t).

    Estimator: for commit decisions (ALLOW/VERIFY/ROUTE/ESCALATE) q_t =
    1 - p_hat_t where p_hat_t is the model's calibrated failure probability
    for the event, clamped to [0.01, 0.99] so the Brier penalty stays
    bounded and bins never degenerate. DENY/ABSTAIN emit no committed
    success forecast, so they are scored at the maximum-entropy q_t = 0.5
    uncertainty; a missing calibrated_probability is treated the same way.
    One value per event; empty input yields [].
    """
    return [_q_for_event(event) for event in events]


def trajectory_proper_score(
    events: Sequence[DecisionEvent], final_success: bool
) -> float:
    """TPS: sum over steps of Brier penalties (q_t - y)^2.

    y = 1 if final_success else 0 is the observed prefix outcome, applied
    to every step: the trajectory outcome is only known at the end, so each
    prefix forecast is scored against it. The per-step Brier penalty is
    strictly proper for the binary outcome, hence the sum is strictly proper
    for the q_t process: a well-calibrated forecaster minimizes it in
    expectation. Lower is better. Returns 0.0 for empty trajectories.
    """
    q = trajectory_success_probs(events)
    if not q:
        return 0.0
    y = 1.0 if final_success else 0.0
    return float(np.sum((np.asarray(q) - y) ** 2))


def tps_report(
    trajectories: Sequence[tuple[Sequence[DecisionEvent], bool]],
) -> dict:
    """Summary of TPS over a batch of (events, final_success) trajectories.

    Returns {"n", "mean_tps", "std_tps", "median_tps", "worst", "best"};
    std_tps is the population standard deviation (0.0 for n < 2), worst is
    the highest (poorest) score and best the lowest. Empty input yields
    all-zero statistics.
    """
    if not trajectories:
        return {
            "n": 0,
            "mean_tps": 0.0,
            "std_tps": 0.0,
            "median_tps": 0.0,
            "worst": 0.0,
            "best": 0.0,
        }
    scores = np.asarray(
        [trajectory_proper_score(events, ok) for events, ok in trajectories]
    )
    return {
        "n": int(scores.size),
        "mean_tps": float(np.mean(scores)),
        "std_tps": float(np.std(scores)),
        "median_tps": float(np.median(scores)),
        "worst": float(np.max(scores)),
        "best": float(np.min(scores)),
    }


def ece_over_trajectories(
    events: Sequence[Sequence[DecisionEvent]],
    final_successes: Sequence[bool],
) -> float:
    """Trajectory-level ECE over final predicted success probabilities.

    For each trajectory the last forecast q_T (from trajectory_success_probs)
    is bucketed into 10 bins and compared against the observed final success
    indicator; ECE = sum_b (n_b/N) * |acc_b - conf_b|. Empty and prediction-
    less trajectories are skipped. Limitation: this is a scalarized,
    resolution-blind summary — it scores only the final forecast and discards
    the shape of the probability path, so trajectories with identical q_T but
    different intermediate risk profiles compare equal here; use
    trajectory_proper_score for the full-resolution view.
    """
    if len(events) != len(final_successes):
        raise ValueError("events and final_successes must have equal length")
    q_t: list[float] = []
    observed: list[bool] = []
    for traj, ok in zip(events, final_successes, strict=True):
        probs = trajectory_success_probs(traj)
        if not probs:
            continue
        q_t.append(probs[-1])
        observed.append(ok)
    if not q_t:
        return 0.0
    n = len(q_t)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(ECE_N_BINS)]
    for q, ok in zip(q_t, observed, strict=True):
        bins[min(int(q * ECE_N_BINS), ECE_N_BINS - 1)].append((q, ok))
    total = 0.0
    for bucket in bins:
        if not bucket:
            continue
        conf = float(np.mean([c for c, _ in bucket]))
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        total += (len(bucket) / n) * abs(acc - conf)
    return float(total)


def _q_for_event(event: DecisionEvent) -> float:
    if event.decision in (DecisionKind.DENY, DecisionKind.ABSTAIN):
        return Q_UNCERTAINTY
    p = event.calibrated_probability
    if p is None:
        return Q_UNCERTAINTY
    return float(np.clip(1.0 - p, Q_CLAMP_MIN, Q_CLAMP_MAX))
