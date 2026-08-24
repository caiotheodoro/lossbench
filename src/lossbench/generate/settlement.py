"""Settlement domain: FX settlement risk with verifiable signals.

Encoding contract (shared by generator and verifier — the verifier reads ONLY
initial_state, never task.gold):

- HERSTATT_EXPOSURE            -> CRITICAL exposure, HITL
- LIMIT_BREACH (exposure >     -> HIGH exposure, HITL
  counterparty_limit)
- COUNTERPARTY_MISMATCH        -> LOW exposure, ESCALATE
- MISSING_CONFIRMATION         -> LOW (days_missing <= 2) or HIGH (> 2)
- DELAYED                      -> LOW (delay_days <= 2) or HIGH (> 2)
- ON_TIME                      -> LOW exposure, MATCH

Verdict rules: ON_TIME -> MATCH; exposure_class HIGH/CRITICAL -> HITL;
otherwise ESCALATE. HITL is required iff exposure_class is HIGH or CRITICAL
and the signal is not ON_TIME. Rule order: HERSTATT > LIMIT_BREACH >
COUNTERPARTY_MISMATCH > MISSING_CONFIRMATION > DELAYED > ON_TIME.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from lossbench.generate.prompt import render_prompt
from lossbench.generate.taxonomy import task_signature
from lossbench.schema import Severity, Task

SIGNALS = (
    "ON_TIME",
    "DELAYED",
    "MISSING_CONFIRMATION",
    "LIMIT_BREACH",
    "COUNTERPARTY_MISMATCH",
    "HERSTATT_EXPOSURE",
)

SEVERITY_BY_SIGNAL: dict[str, Severity] = {
    "ON_TIME": Severity.LOW,
    "DELAYED": Severity.MEDIUM,
    "MISSING_CONFIRMATION": Severity.MEDIUM,
    "LIMIT_BREACH": Severity.HIGH,
    "COUNTERPARTY_MISMATCH": Severity.HIGH,
    "HERSTATT_EXPOSURE": Severity.CRITICAL,
}

_SIGNALS_BY_SEVERITY: dict[str, tuple[str, ...]] = {
    "CRITICAL": ("HERSTATT_EXPOSURE",),
    "HIGH": ("LIMIT_BREACH", "COUNTERPARTY_MISMATCH"),
    "MEDIUM": ("MISSING_CONFIRMATION", "DELAYED"),
    "LOW": ("ON_TIME",),
}

_BICS = ("DEUTDEFF", "CHASUS33", "BARCGB22", "HSBCCNSH", "ITNLUS33")
_NOSTROS = ("CITIUS33", "BOFAUS3N", "HSBCUS33", "JPMCUS33", "BNYMUS33")
_CURRENCY_PAIRS = ("EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP", "USD/CHF")
_T2_GRACE_DAYS = 2


def _parse_amount(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except Exception:
        return None


def _exposure_class_from_signal(
    signal: str,
    detail: dict[str, Any],
    trade: dict[str, Any],
) -> str:
    if signal == "HERSTATT_EXPOSURE":
        return "CRITICAL"
    if signal == "LIMIT_BREACH":
        exposure = _parse_amount(str(trade.get("counterparty_exposure", "")))
        limit = _parse_amount(str(trade.get("counterparty_limit", "")))
        if exposure is not None and limit is not None and exposure > limit:
            return "HIGH"
        return "LOW"
    if signal == "DELAYED" and int(detail.get("delay_days", 0)) > _T2_GRACE_DAYS:
        return "HIGH"
    if signal == "MISSING_CONFIRMATION" and int(detail.get("days_missing", 0)) > _T2_GRACE_DAYS:
        return "HIGH"
    return "LOW"


def compute_verdict_from_state(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Field-derived verdict and exposure class from initial_state only."""
    signal_state = initial_state["settlement_signal"]
    signal = signal_state["signal"]
    detail = signal_state.get("detail", {})
    exposure_class = _exposure_class_from_signal(signal, detail, initial_state["trade"])
    if signal == "ON_TIME":
        return {"verdict": "MATCH", "exception_type": None, "exposure_class": exposure_class}
    if exposure_class in ("HIGH", "CRITICAL"):
        return {"verdict": "HITL", "exception_type": signal, "exposure_class": exposure_class}
    return {"verdict": "ESCALATE", "exception_type": signal, "exposure_class": exposure_class}


def verifier_settlement(task: Task, proposed_outcome: dict[str, Any]) -> bool:
    """True iff proposed_outcome matches the field-derived settlement verdict."""
    expected = compute_verdict_from_state(task.initial_state)
    return all(proposed_outcome.get(key) == value for key, value in expected.items())


def _draw_signal(rng: random.Random, severity_mix: dict[str, float] | None) -> str:
    weights: list[tuple[str, float]] = []
    if severity_mix:
        for sev_value, weight in severity_mix.items():
            pool = _SIGNALS_BY_SEVERITY.get(sev_value.upper(), ())
            if pool:
                weights.append((rng.choice(pool), weight))
    else:
        weights = [(signal, 1.0) for signal in SIGNALS]
    if not weights:
        raise ValueError("severity_mix yields no signal classes")
    total = sum(w for _, w in weights)
    roll = rng.random() * total
    cumulative = 0.0
    for signal, w in weights:
        cumulative += w
        if roll <= cumulative:
            return signal
    return weights[-1][0]


def _make_trade(rng: random.Random, difficulty: float) -> dict[str, str]:
    booking = date(2026, 8, 1) + timedelta(days=rng.randrange(0, 60))
    value_date = booking + timedelta(days=2)
    notional = Decimal(rng.randrange(1000, 5000000, 13)) + Decimal(rng.randrange(0, 100)) / 100
    limit = notional * Decimal(str(round(rng.uniform(1.2, 3.0), 6)))
    exposure = limit * Decimal(str(round(rng.uniform(0.3, 0.85), 6)))
    limit = limit.quantize(Decimal("0.01"))
    exposure = exposure.quantize(Decimal("0.01"))
    return {
        "trade_id": f"TRD{rng.randrange(100000, 999999)}",
        "notional": f"{notional:.2f}",
        "currency_pair": rng.choice(_CURRENCY_PAIRS),
        "value_date": value_date.isoformat(),
        "counterparty": rng.choice(_BICS),
        "nostro": rng.choice(_NOSTROS),
        "booking_date": booking.isoformat(),
        "counterparty_limit": f"{limit:.2f}",
        "counterparty_exposure": f"{exposure:.2f}",
    }


def _days_beyond_t2(rng: random.Random, difficulty: float) -> int:
    if rng.random() < difficulty:
        return rng.choice((2, 3))
    return rng.choice((1, 1, 1, 4, 5, 6))


def _near_miss_bic(rng: random.Random, bic: str) -> str:
    idx = rng.randrange(len(bic))
    replacement = rng.choice([c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != bic[idx]])
    return bic[:idx] + replacement + bic[idx + 1 :]


def _make_signal(
    rng: random.Random,
    signal: str,
    difficulty: float,
    trade: dict[str, str],
) -> dict[str, Any]:
    if signal == "ON_TIME":
        return {
            "signal": signal,
            "detail": {
                "status": "SETTLED",
                "confirmation_received": True,
                "expected_settlement": trade["value_date"],
            },
        }
    if signal == "DELAYED":
        days = _days_beyond_t2(rng, difficulty)
        expected = (date.fromisoformat(trade["value_date"]) + timedelta(days=days)).isoformat()
        return {"signal": signal, "detail": {"delay_days": days, "expected_settlement": expected}}
    if signal == "MISSING_CONFIRMATION":
        days = _days_beyond_t2(rng, difficulty)
        return {
            "signal": signal,
            "detail": {"days_missing": days, "confirmation_sender": trade["counterparty"]},
        }
    if signal == "LIMIT_BREACH":
        limit = _parse_amount(trade["counterparty_limit"]) or Decimal("0")
        if rng.random() < difficulty:
            ratio = rng.uniform(0.001, 0.02)
        else:
            ratio = rng.uniform(0.05, 0.30)
        breach = (limit * Decimal(str(round(ratio, 6)))).quantize(Decimal("0.01"))
        trade["counterparty_exposure"] = f"{limit + breach:.2f}"
        return {
            "signal": signal,
            "detail": {
                "breach_amount": f"{breach:.2f}",
                "limit": trade["counterparty_limit"],
                "exposure": trade["counterparty_exposure"],
            },
        }
    if signal == "COUNTERPARTY_MISMATCH":
        if rng.random() < difficulty:
            instruction = _near_miss_bic(rng, trade["counterparty"])
        else:
            instruction = rng.choice([b for b in _BICS if b != trade["counterparty"]])
        return {
            "signal": signal,
            "detail": {
                "instruction_counterparty": instruction,
                "trade_counterparty": trade["counterparty"],
            },
        }
    if signal == "HERSTATT_EXPOSURE":
        return {
            "signal": signal,
            "detail": {"pvp": False, "principal_legs": 2, "window": "CLOSED"},
        }
    raise ValueError(f"unhandled signal {signal}")


def generate_settlement_task(
    rng: random.Random,
    task_id: str,
    difficulty: float,
    severity_mix: dict[str, float] | None,
    policy_id: str,
    cost_model_ref: str,
) -> Task:
    """Generate one settlement-risk task for a single FX trade."""
    trade = _make_trade(rng, difficulty)
    signal = _draw_signal(rng, severity_mix)
    state = {"trade": trade, "settlement_signal": _make_signal(rng, signal, difficulty, trade)}
    task = Task(
        id=task_id,
        domain="settlement",
        prompt=render_prompt(
            "Assess settlement risk for the FX trade. Decide MATCH (settle as "
            "normal), ESCALATE (route to a senior operator), or HITL (halt and "
            "route to a human). Report the exposure class and, if a signal "
            "applies, name the exception type exactly.",
            state,
            "settlement",
        ),
        initial_state=state,
        available_tools=[
            "check_counterparty_limit",
            "check_settlement_status",
            "classify_exposure",
        ],
        policy_id=policy_id,
        gold=compute_verdict_from_state(state),
        severity=SEVERITY_BY_SIGNAL[signal],
        verifier="verifier_settlement",
        cost_model_ref=cost_model_ref,
        difficulty=round(difficulty, 3),
        seed=rng.randrange(1 << 30),
    )
    task.signature = task_signature(task)
    return task


def generate_settlement_suite_internal(
    seed: int,
    n_tasks: int,
    severity_mix: dict[str, float] | None,
    difficulty: tuple[float, float],
    verifier: Callable[[Task, dict[str, Any]], bool],
    domain = "settlement"
) -> list[Task]:
    """Seeded suite with the verifier-as-oracle self-check loop."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen_signatures: set[str] = set()
    attempts = 0
    max_attempts = n_tasks * 10
    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1
        diff = difficulty[0] + (difficulty[1] - difficulty[0]) * rng.random()
        task_id = f"{domain}:{seed}:{len(tasks)}"
        candidate = generate_settlement_task(
            rng,
            task_id,
            diff,
            severity_mix,
            policy_id="p0",
            # principal_risk is the large-value settlement profile; there has
            # never been a settlement.yaml, so this ref used to raise on load.
            cost_model_ref="principal_risk",
        )
        if not verifier(candidate, candidate.gold):
            continue
        if candidate.signature in seen_signatures:
            continue
        seen_signatures.add(candidate.signature)
        tasks.append(candidate)
    if len(tasks) < n_tasks:
        raise RuntimeError(
            f"generator exhausted retries: {len(tasks)}/{n_tasks} tasks validated"
        )
    return tasks
