"""Reconciliation domain: ledger-vs-statement pairs with verifiable exceptions.

Encoding contract (shared by generator and verifier — the verifier reads ONLY
initial_state, never task.gold):

- statement missing            -> MISSING_MESSAGE
- any field ill-formed         -> FIELD_CORRUPTION
- currency differs             -> FX_CONVERSION_ERROR
- same currency, amount off by
  more than rounding tolerance -> AMOUNT_MISMATCH
- reference already booked     -> DUPLICATE
- beneficiary differs, no token
  overlap                      -> BENEFICIARY_MISMATCH
- beneficiary differs, token
  overlap                      -> PARTIAL_MATCH
- counterparty differs         -> COUNTERPARTY_MISMATCH
- value date differs and (weekend
  or > 2 days past booking)    -> VALUE_DATE_MISMATCH
- otherwise                    -> MATCH

Priority: missing > corruption > amount/FX > duplicate > beneficiary/
counterparty > value-date > match. A task carries exactly one exception.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from lossbench.generate.taxonomy import task_signature
from lossbench.schema import Severity, Task

EXCEPTION_TYPES = (
    "AMOUNT_MISMATCH",
    "FX_CONVERSION_ERROR",
    "BENEFICIARY_MISMATCH",
    "COUNTERPARTY_MISMATCH",
    "VALUE_DATE_MISMATCH",
    "MISSING_MESSAGE",
    "PARTIAL_MATCH",
    "DUPLICATE",
    "FIELD_CORRUPTION",
)

SEVERITY_BY_EXCEPTION: dict[str, Severity] = {
    "AMOUNT_MISMATCH": Severity.HIGH,
    "FX_CONVERSION_ERROR": Severity.HIGH,
    "BENEFICIARY_MISMATCH": Severity.HIGH,
    "COUNTERPARTY_MISMATCH": Severity.HIGH,
    "VALUE_DATE_MISMATCH": Severity.MEDIUM,
    "MISSING_MESSAGE": Severity.MEDIUM,
    "PARTIAL_MATCH": Severity.MEDIUM,
    "DUPLICATE": Severity.LOW,
    "FIELD_CORRUPTION": Severity.LOW,
}

_BICS = ("DEUTDEFF", "CHASUS33", "BARCGB22", "HSBCCNSH", "ITNLUS33")
_BENEFICIARIES = (
    "ACME INDUSTRIES GMBH",
    "NORTHWIND TRADING LTD",
    "BLUE RIDGE ENERGY LLC",
    "GLOBAL MARITIME SA",
    "PACIFIC SUN CORP",
)
_CURRENCIES = ("USD", "EUR", "GBP", "JPY")
_TOLERANCE = Decimal("0.01")


def _fmt_amount(value: Decimal) -> str:
    return f"{value:.2f}"


def _record(
    rng: random.Random,
    reference: str,
    amount: Decimal,
    currency: str,
    counterparty: str,
    beneficiary: str,
    value_date: date,
    booking_date: date,
) -> dict[str, str]:
    return {
        "reference": reference,
        "amount": _fmt_amount(amount),
        "currency": currency,
        "counterparty": counterparty,
        "beneficiary": beneficiary,
        "value_date": value_date.isoformat(),
        "booking_date": booking_date.isoformat(),
    }


def _parse_amount(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except Exception:
        return None


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in a.replace(",", "").upper().split() if t}
    tb = {t for t in b.replace(",", "").upper().split() if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def compute_verdict_from_state(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Independent verdict computation from initial_state fields only."""
    statement = initial_state.get("statement")
    if statement is None:
        return {"verdict": "EXCEPTION", "exception_type": "MISSING_MESSAGE"}
    ledger = initial_state["ledger"]

    ill_formed = False
    for side in (ledger, statement):
        for key in ("reference", "amount", "currency", "counterparty", "beneficiary"):
            value = side.get(key)
            if value is None or not isinstance(value, str) or not value.strip():
                ill_formed = True
        if _parse_amount(side.get("amount", "")) is None:
            ill_formed = True
    if ill_formed:
        return {"verdict": "EXCEPTION", "exception_type": "FIELD_CORRUPTION"}

    if ledger["currency"] != statement["currency"]:
        return {"verdict": "EXCEPTION", "exception_type": "FX_CONVERSION_ERROR"}

    diff = abs(_parse_amount(ledger["amount"]) - _parse_amount(statement["amount"]))
    if diff > _TOLERANCE:
        return {"verdict": "EXCEPTION", "exception_type": "AMOUNT_MISMATCH"}

    prior = set(initial_state.get("prior_bookings", []))
    if statement["reference"] in prior:
        return {"verdict": "EXCEPTION", "exception_type": "DUPLICATE"}

    if ledger["beneficiary"] != statement["beneficiary"]:
        if _token_overlap(ledger["beneficiary"], statement["beneficiary"]) > 0.0:
            return {"verdict": "EXCEPTION", "exception_type": "PARTIAL_MATCH"}
        return {"verdict": "EXCEPTION", "exception_type": "BENEFICIARY_MISMATCH"}

    if ledger["counterparty"] != statement["counterparty"]:
        return {"verdict": "EXCEPTION", "exception_type": "COUNTERPARTY_MISMATCH"}

    if ledger["value_date"] != statement["value_date"]:
        vd = date.fromisoformat(statement["value_date"])
        bd = date.fromisoformat(ledger["booking_date"])
        if _is_weekend(vd) or (vd - bd).days > 2:
            return {"verdict": "EXCEPTION", "exception_type": "VALUE_DATE_MISMATCH"}

    return {"verdict": "MATCH", "exception_type": None}


def verifier_reconciliation(task: Task, proposed_outcome: dict[str, Any]) -> bool:
    """True iff proposed_outcome matches the field-derived verdict."""
    expected = compute_verdict_from_state(task.initial_state)
    if proposed_outcome.get("verdict") != expected["verdict"]:
        return False
    if proposed_outcome.get("exception_type") != expected["exception_type"]:
        return False
    return True


def _draw_exception_type(rng: random.Random, severity_mix: dict[str, float]) -> str:
    weights: list[tuple[str, float]] = []
    if severity_mix:
        groups: dict[str, list[str]] = {}
        for exc in EXCEPTION_TYPES:
            groups.setdefault(SEVERITY_BY_EXCEPTION[exc].value, []).append(exc)
        for sev_value, weight in severity_mix.items():
            pool = groups.get(sev_value.upper(), [])
            if pool:
                weights.append((rng.choice(pool), weight))
    else:
        weights = [(exc, 1.0) for exc in EXCEPTION_TYPES]
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


def _make_clean(
    rng: random.Random,
    difficulty: float,
) -> tuple[dict[str, Any], str, str]:
    """Draw a clean pair; returns (initial_state, ledger_ref, booking_date)."""
    reference = f"REF{rng.randrange(100000, 999999)}"
    currency = rng.choice(_CURRENCIES)
    booking = date(2026, 8, 1) + timedelta(days=rng.randrange(0, 60))
    value_date = booking + timedelta(days=rng.randrange(1, 3))
    if difficulty > 0.5 and not _is_weekend(value_date):
        value_date = booking + timedelta(days=1)
    amount = Decimal(rng.randrange(100, 100000, 13)) + Decimal(rng.randrange(0, 100)) / 100
    ledger = _record(
        rng, reference, amount, currency,
        rng.choice(_BICS), rng.choice(_BENEFICIARIES),
        value_date, booking,
    )
    statement = dict(ledger)
    return {"ledger": ledger, "statement": statement}, reference, booking.isoformat()


def _near_miss_amount(rng: random.Random, amount: Decimal) -> Decimal:
    return amount + Decimal(rng.choice(("0.005", "-0.005", "0.00")))


def _apply_exception(
    rng: random.Random,
    state: dict[str, Any],
    exc_type: str,
    difficulty: float,
    booking_date: str,
) -> dict[str, Any]:
    ledger = state["ledger"]
    statement = state["statement"]

    if exc_type == "MISSING_MESSAGE":
        state["statement"] = None
        return state

    if exc_type == "FIELD_CORRUPTION":
        corrupt = statement if rng.random() < 0.5 else ledger
        key = rng.choice(("amount", "reference", "beneficiary"))
        if key == "amount":
            corrupt[key] = "12,3a4.55"
        elif key == "reference":
            corrupt[key] = ""
        else:
            corrupt[key] = "X"
        return state

    if exc_type == "FX_CONVERSION_ERROR":
        other = rng.choice([c for c in _CURRENCIES if c != ledger["currency"]])
        statement["currency"] = other
        return state

    if exc_type == "AMOUNT_MISMATCH":
        statement["amount"] = _fmt_amount(_parse_amount(statement["amount"]) + Decimal("12.37"))
        return state

    if exc_type == "DUPLICATE":
        state["prior_bookings"] = [ledger["reference"]]
        return state

    if exc_type == "BENEFICIARY_MISMATCH":
        statement["beneficiary"] = "TOTALLY DIFFERENT ENTITY OY"
        return state

    if exc_type == "PARTIAL_MATCH":
        base = ledger["beneficiary"]
        tokens = [t for t in base.replace(",", "").upper().split() if t]
        statement["beneficiary"] = " ".join(tokens[:-1] + ["INC"])
        return state

    if exc_type == "COUNTERPARTY_MISMATCH":
        statement["counterparty"] = rng.choice([b for b in _BICS if b != ledger["counterparty"]])
        return state

    if exc_type == "VALUE_DATE_MISMATCH":
        bd = date.fromisoformat(booking_date)
        statement["value_date"] = (bd + timedelta(days=5)).isoformat()
        return state

    raise ValueError(f"unhandled exception type {exc_type}")


def generate_reconciliation_task(
    rng: random.Random,
    task_id: str,
    difficulty: float,
    severity_mix: dict[str, float] | None,
    policy_id: str,
    cost_model_ref: str,
) -> Task:
    """Generate one task: clean (MATCH) or a single injected exception."""
    clean = rng.random() < 0.45
    state, reference, booking_date = _make_clean(rng, difficulty)

    if clean:
        amount = _parse_amount(state["ledger"]["amount"])
        state["ledger"]["amount"] = _fmt_amount(_near_miss_amount(rng, amount))
        exception_type = None
        severity = None
    else:
        exc_type = _draw_exception_type(rng, severity_mix)
        _apply_exception(rng, state, exc_type, difficulty, booking_date)
        exception_type = exc_type
        severity = SEVERITY_BY_EXCEPTION[exc_type]

    gold = {
        "verdict": "MATCH" if clean else "EXCEPTION",
        "exception_type": exception_type,
    }
    task = Task(
        id=task_id,
        domain="reconciliation",
        prompt=(
            "Compare the ledger record and the counterparty statement. "
            "Classify the pair as MATCH or EXCEPTION. If EXCEPTION, name the "
            "exception type exactly."
        ),
        initial_state=state,
        available_tools=["lookup_prior_bookings", "classify_pair"],
        policy_id=policy_id,
        gold=gold,
        severity=severity if severity is not None else Severity.LOW,
        verifier="verifier_reconciliation",
        cost_model_ref=cost_model_ref,
        difficulty=round(difficulty, 3),
        seed=rng.randrange(1 << 30),
    )
    task.signature = task_signature(task)
    return task


def generate_suite_internal(
    seed: int,
    n_tasks: int,
    severity_mix: dict[str, float] | None,
    difficulty: tuple[float, float],
    verifier: Callable[[Task, dict[str, Any]], bool],
    domain = "reconciliation"
) -> list[Task]:
    """Seeded suite with the verifier-as-oracle self-check loop."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    attempts = 0
    max_attempts = n_tasks * 10
    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1
        diff = difficulty[0] + (difficulty[1] - difficulty[0]) * rng.random()
        task_id = f"{domain}:{seed}:{len(tasks)}"
        candidate = generate_reconciliation_task(
            rng, task_id, diff, severity_mix, policy_id="p0", cost_model_ref="reconciliation"
        )
        outcome = {
            "verdict": candidate.gold["verdict"],
            "exception_type": candidate.gold["exception_type"],
        }
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
