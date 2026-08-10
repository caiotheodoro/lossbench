"""Payment-repair domain: failed-STP payments with verifiable repair decisions.

Encoding contract (shared by generator and verifier — the verifier reads ONLY
initial_state, never task.gold):

- FRAUD_HOLD                            -> EXCEPTION / hold_hitl / unsafe
- creditor or creditor_name absent      -> HITL / hold_hitl / unsafe
- AMOUNT_MISMATCH, deviation <= 2%      -> REPAIR_SAFE / amend_amount / safe
- amount deviation > 2%                 -> HITL / hold_hitl / unsafe
- BENEFICIARY_MISMATCH, overlap >= 0.5  -> REPAIR_SAFE / amend_beneficiary / safe
- beneficiary overlap < 0.5             -> HITL / hold_hitl / unsafe
- DUPLICATE_ALERT                       -> HITL / hold_hitl / unsafe
- VALUE_DATE                            -> REPAIR_SAFE / resubmit / safe
- otherwise                             -> REPAIR_SAFE / approve / safe

Priority: fraud > missing creditor > amount > 2% > beneficiary overlap < 0.5 >
duplicate > value date > safe approve. A task carries exactly one signal.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from lossbench.generate.taxonomy import task_signature
from lossbench.schema import Severity, Task

EXCEPTION_SIGNALS = (
    "RETURNED",
    "REJECTED",
    "DUPLICATE_ALERT",
    "AMOUNT_MISMATCH",
    "MISSING_FIELD",
    "BENEFICIARY_MISMATCH",
    "VALUE_DATE",
    "FRAUD_HOLD",
)

SEVERITY_BY_SIGNAL: dict[str, Severity] = {
    "FRAUD_HOLD": Severity.CRITICAL,
    "AMOUNT_MISMATCH": Severity.HIGH,
    "BENEFICIARY_MISMATCH": Severity.HIGH,
    "RETURNED": Severity.MEDIUM,
    "REJECTED": Severity.MEDIUM,
    "VALUE_DATE": Severity.MEDIUM,
    "DUPLICATE_ALERT": Severity.LOW,
    "MISSING_FIELD": Severity.LOW,
}

REPAIR_VERBS = (
    "approve",
    "amend_amount",
    "amend_beneficiary",
    "resubmit",
    "hold_hitl",
    "reject",
)

_BICS = ("DEUTDEFF", "CHASUS33", "BARCGB22", "HSBCCNSH", "ITNLUS33")
_NAMES = (
    "ACME INDUSTRIES GMBH",
    "NORTHWIND TRADING LTD",
    "BLUE RIDGE ENERGY LLC",
    "GLOBAL MARITIME SA",
    "PACIFIC SUN CORP",
)
_CURRENCIES = ("USD", "EUR", "GBP", "JPY")
_INSTRUCTION_TYPES = ("pacs.008", "pacs.009", "MT103", "Fedwire")
_RETURN_REASONS = (
    "beneficiary_account_closed",
    "beneficiary_iban_invalid",
    "recipient_bank_unreachable",
)
_REJECT_REASONS = (
    "instructing_party_restriction",
    "beneficiary_name_regulatory_mismatch",
    "compliance_rejection",
)
_WATCHLISTS = ("OFAC_SDN", "EU_CONSOLIDATED", "UN_SC")
_NEAR_MISS_SUFFIXES = ("HOLDING", "AG", "BV", "PARTNERS")
_NEAR_MISS_TOLERANCE = 0.02
_NEAR_MISS_AMOUNT_FRACTION = Decimal("0.01")
_HARD_AMOUNT_FRACTION = Decimal("0.12")
_OVERLAP_THRESHOLD = 0.5


def _fmt_amount(value: Decimal) -> str:
    return f"{value:.2f}"


def _parse_amount(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except Exception:
        return None


def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in a.replace(",", "").upper().split() if t}
    tb = {t for t in b.replace(",", "").upper().split() if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _amount_deviation(payment: dict[str, Any], detail: dict[str, Any]) -> float:
    paid = _parse_amount(str(payment.get("amount", "")))
    expected = _parse_amount(str(detail.get("source_amount") or payment.get("amount", "")))
    if paid is None or expected is None or expected <= 0:
        return 0.0
    return float(abs(paid - expected) / expected)


def _beneficiary_overlap(payment: dict[str, Any], detail: dict[str, Any]) -> float:
    source = detail.get("source_creditor_name") or payment.get("creditor_name", "")
    return _token_overlap(str(payment.get("creditor_name", "")), str(source))


def compute_repair_decision_from_state(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Independent repair decision computed from initial_state fields only."""
    payment = initial_state["payment"]
    signal = initial_state["exception_signal"]["signal"]
    detail = initial_state.get("exception_signal", {}).get("detail", {})
    deviation = _amount_deviation(payment, detail)
    overlap = _beneficiary_overlap(payment, detail)
    creditor_missing = not str(payment.get("creditor", "")).strip()
    creditor_name_missing = not str(payment.get("creditor_name", "")).strip()

    if signal == "FRAUD_HOLD":
        return {
            "verdict": "EXCEPTION",
            "exception_type": signal,
            "repair_safe": False,
            "repair": "hold_hitl",
        }
    if creditor_missing or creditor_name_missing:
        return {
            "verdict": "HITL",
            "exception_type": signal,
            "repair_safe": False,
            "repair": "hold_hitl",
        }
    if signal == "AMOUNT_MISMATCH" and deviation <= _NEAR_MISS_TOLERANCE:
        return {
            "verdict": "REPAIR_SAFE",
            "exception_type": signal,
            "repair_safe": True,
            "repair": "amend_amount",
        }
    if deviation > _NEAR_MISS_TOLERANCE:
        return {
            "verdict": "HITL",
            "exception_type": signal,
            "repair_safe": False,
            "repair": "hold_hitl",
        }
    if signal == "BENEFICIARY_MISMATCH" and overlap >= _OVERLAP_THRESHOLD:
        return {
            "verdict": "REPAIR_SAFE",
            "exception_type": signal,
            "repair_safe": True,
            "repair": "amend_beneficiary",
        }
    if overlap < _OVERLAP_THRESHOLD:
        return {
            "verdict": "HITL",
            "exception_type": signal,
            "repair_safe": False,
            "repair": "hold_hitl",
        }
    if signal == "DUPLICATE_ALERT":
        return {
            "verdict": "HITL",
            "exception_type": signal,
            "repair_safe": False,
            "repair": "hold_hitl",
        }
    if signal == "VALUE_DATE":
        return {
            "verdict": "REPAIR_SAFE",
            "exception_type": signal,
            "repair_safe": True,
            "repair": "resubmit",
        }
    return {
        "verdict": "REPAIR_SAFE",
        "exception_type": signal,
        "repair_safe": True,
        "repair": "approve",
    }


def verifier_payment_repair(task: Task, proposed_outcome: dict[str, Any]) -> bool:
    """True iff proposed_outcome matches the field-derived repair decision."""
    expected = compute_repair_decision_from_state(task.initial_state)
    for key in ("verdict", "exception_type", "repair", "repair_safe"):
        if proposed_outcome.get(key) != expected[key]:
            return False
    return True


def _draw_signal(rng: random.Random, severity_mix: dict[str, float]) -> str:
    weights: list[tuple[str, float]] = []
    if severity_mix:
        groups: dict[str, list[str]] = {}
        for exc in EXCEPTION_SIGNALS:
            groups.setdefault(SEVERITY_BY_SIGNAL[exc].value, []).append(exc)
        for sev_value, weight in severity_mix.items():
            pool = groups.get(sev_value.upper(), [])
            if pool:
                weights.append((rng.choice(pool), weight))
    else:
        weights = [(exc, 1.0) for exc in EXCEPTION_SIGNALS]
    if not weights:
        raise ValueError("severity_mix yields no exception classes")
    total = sum(w for _, w in weights)
    roll = rng.random() * total
    cumulative = 0.0
    for exc, w in weights:
        cumulative += w
        if roll <= cumulative:
            return exc
    return weights[-1][0]


def _make_payment(rng: random.Random) -> dict[str, Any]:
    booking = date(2026, 8, 1) + timedelta(days=rng.randrange(0, 60))
    amount = Decimal(rng.randrange(100, 100000, 13)) + Decimal(rng.randrange(0, 100)) / 100
    return {
        "reference": f"PMT{rng.randrange(100000, 999999)}",
        "amount": _fmt_amount(amount),
        "currency": rng.choice(_CURRENCIES),
        "debtor": rng.choice(_BICS),
        "creditor": rng.choice(_BICS),
        "debtor_name": rng.choice(_NAMES),
        "creditor_name": rng.choice(_NAMES),
        "value_date": (booking + timedelta(days=rng.randrange(1, 3))).isoformat(),
        "instruction_type": rng.choice(_INSTRUCTION_TYPES),
    }


def _apply_signal(
    rng: random.Random,
    payment: dict[str, Any],
    signal: str,
    difficulty: float,
) -> dict[str, Any]:
    if signal == "AMOUNT_MISMATCH":
        source = _parse_amount(payment["amount"])
        fraction = _NEAR_MISS_AMOUNT_FRACTION if difficulty > 0.5 else _HARD_AMOUNT_FRACTION
        delta = (source * fraction).quantize(Decimal("0.01"))
        payment["amount"] = _fmt_amount(source + delta)
        return {"source_amount": _fmt_amount(source)}

    if signal == "BENEFICIARY_MISMATCH":
        source = payment["creditor_name"]
        if difficulty > 0.5:
            suffix = rng.choice([s for s in _NEAR_MISS_SUFFIXES if s not in source.upper()])
            payment["creditor_name"] = f"{source} {suffix}"
        else:
            payment["creditor_name"] = "TOTALLY DIFFERENT ENTITY OY"
        return {"source_creditor_name": source}

    if signal == "MISSING_FIELD":
        field = rng.choice(("creditor_name", "instruction_type"))
        if field == "creditor_name":
            source = payment["creditor_name"]
            payment["creditor_name"] = ""
            return {"missing_field": field, "source_creditor_name": source}
        payment["instruction_type"] = ""
        return {"missing_field": field}

    if signal == "VALUE_DATE":
        source = payment["value_date"]
        offset = timedelta(days=1) if difficulty > 0.5 else timedelta(days=5)
        payment["value_date"] = (date.fromisoformat(source) + offset).isoformat()
        return {"source_value_date": source}

    if signal == "DUPLICATE_ALERT":
        return {
            "prior_reference": f"PMT{rng.randrange(100000, 999999)}",
            "match_score": round(rng.uniform(0.92, 0.99), 4),
        }

    if signal == "FRAUD_HOLD":
        return {
            "watchlist_hit": rng.choice(_WATCHLISTS),
            "risk_score": rng.randrange(85, 100),
        }

    if signal == "RETURNED":
        return {"reason": rng.choice(_RETURN_REASONS)}

    return {"reason": rng.choice(_REJECT_REASONS)}


def generate_payment_repair_task(
    rng: random.Random,
    task_id: str,
    difficulty: float,
    severity_mix: dict[str, float] | None,
    policy_id: str,
    cost_model_ref: str,
) -> Task:
    """Generate one payment-repair task: a clean payment plus one exception signal."""
    payment = _make_payment(rng)
    signal = _draw_signal(rng, severity_mix)
    detail = _apply_signal(rng, payment, signal, difficulty)
    state = {"payment": payment, "exception_signal": {"signal": signal, "detail": detail}}
    decision = compute_repair_decision_from_state(state)
    task = Task(
        id=task_id,
        domain="payment_repair",
        prompt=(
            "An ISO 20022 payment instruction failed straight-through processing "
            "and landed in the exception/repair queue. Classify the exception "
            "signal exactly, decide whether automated repair is safe, and choose "
            "the repair verb from: approve, amend_amount, amend_beneficiary, "
            "resubmit, hold_hitl, reject."
        ),
        initial_state=state,
        available_tools=["classify_exception", "propose_repair", "hold_for_human_review"],
        policy_id=policy_id,
        gold=decision,
        severity=SEVERITY_BY_SIGNAL[signal],
        verifier="verifier_payment_repair",
        cost_model_ref=cost_model_ref,
        difficulty=round(difficulty, 3),
        seed=rng.randrange(1 << 30),
    )
    task.signature = task_signature(task)
    return task


def generate_payment_repair_suite_internal(
    seed: int,
    n_tasks: int,
    severity_mix: dict[str, float] | None,
    difficulty: tuple[float, float],
    verifier: Callable[[Task, dict[str, Any]], bool],
    domain = "payment_repair"
) -> list[Task]:
    """Seeded payment-repair suite with the verifier-as-oracle self-check loop."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    attempts = 0
    max_attempts = n_tasks * 10
    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1
        diff = difficulty[0] + (difficulty[1] - difficulty[0]) * rng.random()
        task_id = f"{domain}:{seed}:{len(tasks)}"
        candidate = generate_payment_repair_task(
            rng, task_id, diff, severity_mix, policy_id="p0", cost_model_ref="principal_risk"
        )
        outcome = dict(candidate.gold)
        if not verifier(candidate, outcome):
            continue
        if candidate.signature in seen_ids:
            continue
        seen_ids.add(candidate.signature)
        tasks.append(candidate)
    if len(tasks) < n_tasks:
        raise RuntimeError(
            f"generator exhausted retries: {len(tasks)}/{n_tasks} tasks validated"
        )
    return tasks
