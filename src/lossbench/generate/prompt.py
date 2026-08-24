"""Model-facing prompt rendering: instruction + input data + output contract.

A task's `prompt` must be answerable on its own. Before this module the
prompt carried only the instruction, so `initial_state` never reached the
model and every real run scored zero for reasons unrelated to the model.

The rendered prompt is part of the task, so it is covered by
`task_signature` and by the byte-identical determinism gate.
"""

from __future__ import annotations

import json
from typing import Any

_RECONCILIATION_EXCEPTIONS = (
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

_PAYMENT_SIGNALS = (
    "FRAUD_HOLD",
    "AMOUNT_MISMATCH",
    "BENEFICIARY_MISMATCH",
    "RETURNED",
    "REJECTED",
    "VALUE_DATE",
    "DUPLICATE_ALERT",
    "MISSING_FIELD",
)

_REPAIR_VERBS = (
    "approve",
    "amend_amount",
    "amend_beneficiary",
    "resubmit",
    "hold_hitl",
    "reject",
)

_SETTLEMENT_SIGNALS = (
    "DELAYED",
    "MISSING_CONFIRMATION",
    "LIMIT_BREACH",
    "COUNTERPARTY_MISMATCH",
    "HERSTATT_EXPOSURE",
)

# key -> allowed values, or None when the value is free-form (e.g. a bool).
OUTPUT_SCHEMAS: dict[str, dict[str, tuple[str, ...] | None]] = {
    "reconciliation": {
        "verdict": ("MATCH", "EXCEPTION"),
        "exception_type": _RECONCILIATION_EXCEPTIONS,
        "confidence": None,
    },
    "payment_repair": {
        "verdict": ("REPAIR_SAFE", "EXCEPTION", "HITL"),
        "exception_type": _PAYMENT_SIGNALS,
        "repair_safe": None,
        "repair": _REPAIR_VERBS,
        "confidence": None,
    },
    "settlement": {
        "verdict": ("MATCH", "ESCALATE", "HITL"),
        "exception_type": _SETTLEMENT_SIGNALS,
        "exposure_class": ("LOW", "HIGH", "CRITICAL"),
        "confidence": None,
    },
}

# Decision rules the domain verifier applies. These are policy, not model
# knowledge: without them a model is scored on guessing an unpublished
# convention rather than on applying a stated one. Kept in the same order
# the verifier evaluates them, since the rules are first-match.
POLICY_RULES: dict[str, tuple[str, ...]] = {
    "reconciliation": (
        "If every field agrees between the ledger record and the statement, "
        "verdict is MATCH and exception_type is null.",
        "Otherwise verdict is EXCEPTION and exception_type names the single "
        "discrepancy class.",
    ),
    "payment_repair": (
        "Apply the first rule that matches.",
        "1. signal FRAUD_HOLD -> EXCEPTION, repair_safe false, repair hold_hitl.",
        "2. payment.creditor or payment.creditor_name blank or missing -> HITL, "
        "repair_safe false, repair hold_hitl.",
        "3. signal AMOUNT_MISMATCH and the amount differs from "
        "detail.source_amount by 2% or less -> REPAIR_SAFE, repair_safe true, "
        "repair amend_amount.",
        "4. the amount differs by more than 2% -> HITL, repair_safe false, "
        "repair hold_hitl.",
        "5. signal BENEFICIARY_MISMATCH and creditor_name shares at least half "
        "its word tokens with detail.source_creditor_name -> REPAIR_SAFE, "
        "repair_safe true, repair amend_beneficiary.",
        "6. that token overlap is below half -> HITL, repair_safe false, "
        "repair hold_hitl.",
        "7. signal DUPLICATE_ALERT -> HITL, repair_safe false, repair hold_hitl.",
        "8. signal VALUE_DATE -> REPAIR_SAFE, repair_safe true, repair resubmit.",
        "9. otherwise -> REPAIR_SAFE, repair_safe true, repair approve.",
        "exception_type is always the signal named in the input.",
    ),
    "settlement": (
        "First derive exposure_class:",
        "- signal HERSTATT_EXPOSURE -> CRITICAL.",
        "- signal LIMIT_BREACH -> HIGH when trade.counterparty_exposure exceeds "
        "trade.counterparty_limit, otherwise LOW.",
        "- signal DELAYED -> HIGH when detail.delay_days is more than 2, "
        "otherwise LOW.",
        "- signal MISSING_CONFIRMATION -> HIGH when detail.days_missing is more "
        "than 2, otherwise LOW.",
        "- anything else -> LOW.",
        "Then the verdict:",
        "- signal ON_TIME -> MATCH with exception_type null.",
        "- exposure_class HIGH or CRITICAL -> HITL.",
        "- otherwise -> ESCALATE.",
        "exception_type is the signal named in the input, except for ON_TIME.",
    ),
}

_VALUE_HINTS: dict[str, str] = {
    "repair_safe": "true or false",
    "confidence": "number from 0 to 1, your probability that this answer is correct",
}


def _field_line(key: str, allowed: tuple[str, ...] | None) -> str:
    if allowed is None:
        return f'  "{key}": {_VALUE_HINTS.get(key, "value")}'
    return f'  "{key}": one of {", ".join(allowed)} (or null)'


def schema_block(domain: str) -> str:
    """Human- and model-readable description of the required JSON object."""
    schema = OUTPUT_SCHEMAS.get(domain)
    if schema is None:
        raise ValueError(f"unknown domain '{domain}'; expected one of {tuple(OUTPUT_SCHEMAS)}")
    fields = "\n".join(_field_line(key, allowed) for key, allowed in schema.items())
    return "{\n" + fields + "\n}"


def render_prompt(instruction: str, initial_state: dict[str, Any], domain: str) -> str:
    """Compose instruction, the task's input data, and the output contract.

    Deterministic for a given (instruction, initial_state, domain): the
    state is serialized with sorted keys, so dict ordering cannot change
    the rendered bytes or the resulting task signature.
    """
    schema = schema_block(domain)
    body = json.dumps(initial_state, indent=2, sort_keys=True, default=str)
    rules = "\n".join(POLICY_RULES[domain])
    return (
        f"{instruction}\n\n"
        f"RULES:\n{rules}\n\n"
        f"INPUT:\n{body}\n\n"
        "Answer with exactly one JSON object and no other text. "
        "Use null where a field does not apply. confidence is your own "
        "calibrated probability that the rest of the object is correct; it is "
        "measured for calibration, never graded against the answer key.\n"
        f"{schema}"
    )


__all__ = ["OUTPUT_SCHEMAS", "POLICY_RULES", "render_prompt", "schema_block"]
