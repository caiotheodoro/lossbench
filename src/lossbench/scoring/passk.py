"""Outcome-verified pass@k / pass^k and severity-corrected scoring.

Pure functions over trial outcome matrices; no I/O. pass@k measures
best-of-k capability, pass^k measures reliability (all k trials must
succeed), and severity_corrected_passk_report combines both with
failure-cost-weighted task credit.
"""

from __future__ import annotations

from collections.abc import Sequence

from lossbench.schema import CostProfile, Severity


def outcome_verified_pass_at_k(trials: Sequence[Sequence[bool]], k: int) -> float:
    """pass@k over k independent trials per task.

    Fraction of tasks with at least one successful trial among the first k
    trials, where trials[task][trial_index] is the outcome of trial_index
    for task. Trials are outcome-verified booleans: a success is only
    credited when the recorded outcome reproduces the gold state. Tasks
    with fewer than k trials are scored over the trials they have. Empty
    input yields 0.0; k < 1 yields 0.0 (no trial lies in the first k).
    """
    if not trials or k < 1:
        return 0.0
    passed = sum(1 for task in trials if any(task[:k]))
    return passed / len(trials)


def pass_k_reliability(trials: Sequence[Sequence[bool]], k: int) -> float:
    """pass^k: all k trials must succeed for the task to pass.

    Reliability analogue of pass@k (best-of-k): a task passes only if it
    has at least k trials and every one of the first k succeeds, so a
    single failure within the first k trials fails the task. Empty input
    yields 0.0; k < 1 yields 0.0.
    """
    if not trials or k < 1:
        return 0.0
    passed = sum(1 for task in trials if len(task) >= k and all(task[:k]))
    return passed / len(trials)


def severity_corrected_passk_report(
    trials: Sequence[Sequence[bool]],
    severities: Sequence[Severity],
    profile: CostProfile,
    k: int,
) -> dict:
    """pass@k / pass^k with failure-cost-weighted credit per task.

    Returns {"pass@k", "pass^k", "severity_weighted_passk"}. The weighted
    score is

        severity_weighted_passk = sum_i [ pass_i * K(sev_i) ] / sum_i [ K(sev_i) ]

    where pass_i = 1 if task i has any successful trial among the first k
    (pass@k per task) else 0, and K(sev_i) = profile.cost(sev_i). Dividing
    numerator and denominator by K_max = max_i K(sev_i) leaves the ratio
    unchanged, so task i's credit weight is w_i = K(sev_i)/K_max in [0, 1]:
    credit is proportional to the business cost of failing the task.
    Failing a HIGH task forfeits a larger weight than failing a LOW task,
    so high-severity failures depress the score more; passing
    high-severity tasks earns correspondingly more credit. Empty trials
    yield all-zero fields; a length mismatch between trials and
    severities raises ValueError.
    """
    if len(trials) != len(severities):
        raise ValueError("trials and severities must have equal length")
    if not trials:
        return {
            "pass@k": 0.0,
            "pass^k": 0.0,
            "severity_weighted_passk": 0.0,
        }
    weights = [profile.cost(sev) for sev in severities]
    total = sum(weights)
    if total == 0.0:
        weighted = 0.0
    else:
        per_task = [any(task[:k]) for task in trials]
        weighted = sum(
            (1.0 if ok else 0.0) * w for ok, w in zip(per_task, weights, strict=True)
        ) / total
    return {
        "pass@k": outcome_verified_pass_at_k(trials, k),
        "pass^k": pass_k_reliability(trials, k),
        "severity_weighted_passk": weighted,
    }
