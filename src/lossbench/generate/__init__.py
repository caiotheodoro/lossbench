"""Seeded, verifier-validated task generation for finance back-office domains.

The generator loop guarantees 100% verifier agreement: every candidate task
runs through its domain verifier before acceptance, and failed draws are
rejected and re-drawn deterministically from the same seeded stream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lossbench.generate.payment_repair import (
    generate_payment_repair_suite_internal,
    verifier_payment_repair,
)
from lossbench.generate.reconciliation import (
    generate_suite_internal as _generate_reconciliation_suite,
)
from lossbench.generate.reconciliation import (
    verifier_reconciliation,
)
from lossbench.generate.settlement import (
    generate_settlement_suite_internal,
    verifier_settlement,
)
from lossbench.generate.taxonomy import task_signature
from lossbench.schema import Task

DOMAINS = ("reconciliation", "payment_repair", "settlement")

_VERIFIERS: dict[str, Callable[[Task, dict[str, Any]], bool]] = {
    "reconciliation": verifier_reconciliation,
    "payment_repair": verifier_payment_repair,
    "settlement": verifier_settlement,
}

_SUITE_GENERATORS: dict[str, Callable[..., list[Task]]] = {
    "reconciliation": _generate_reconciliation_suite,
    "payment_repair": generate_payment_repair_suite_internal,
    "settlement": generate_settlement_suite_internal,
}


def generate_suite(
    domain: str,
    seed: int,
    n_tasks: int,
    severity_mix: dict[str, float] | None = None,
    difficulty: tuple[float, float] = (0.0, 1.0),
    verifier: Callable[[Task, dict[str, Any]], bool] | None = None,
) -> list[Task]:
    """Generate `n_tasks` deterministic, verifier-validated tasks.

    Same seed => byte-identical Task list (compared via model_dump_json).
    severity_mix maps Severity.value -> weight; observed mix honors the
    request within +/-5 percentage points. difficulty is (min, max) and
    scales near-miss adversarial generation.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain '{domain}'; expected one of {DOMAINS}")
    generator = _SUITE_GENERATORS[domain]
    return generator(
        seed=seed,
        n_tasks=n_tasks,
        severity_mix=severity_mix,
        difficulty=difficulty,
        verifier=verifier or _VERIFIERS[domain],
    )


def verifier_for(domain: str) -> Callable[[Task, dict[str, Any]], bool]:
    """Domain verifier registry lookup."""
    try:
        return _VERIFIERS[domain]
    except KeyError as exc:
        raise ValueError(f"no verifier registered for domain '{domain}'") from exc


__all__ = [
    "DOMAINS",
    "generate_suite",
    "task_signature",
    "verifier_for",
]
