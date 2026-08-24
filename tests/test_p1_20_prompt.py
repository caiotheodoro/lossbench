"""Model-facing prompt contract: data in, schema declared, parseable answer out."""

from __future__ import annotations

import json

import pytest

from lossbench.generate import DOMAINS, generate_suite
from lossbench.generate.prompt import OUTPUT_SCHEMAS, render_prompt


def _scalars(value: object) -> list[str]:
    """Every leaf value in a nested structure, rendered the way JSON writes it."""
    if isinstance(value, dict):
        return [s for v in value.values() for s in _scalars(v)]
    if isinstance(value, list):
        return [s for v in value for s in _scalars(v)]
    return [json.dumps(value, default=str)]


def test_every_domain_has_a_schema():
    assert set(OUTPUT_SCHEMAS) == set(DOMAINS)


@pytest.mark.parametrize("domain", DOMAINS)
def test_schema_covers_every_gold_key(domain):
    """The advertised schema must name every key the verifier compares."""
    schema_keys = set(OUTPUT_SCHEMAS[domain])
    for task in generate_suite(domain, seed=777, n_tasks=40):
        assert set(task.gold) <= schema_keys, f"{domain}: gold has keys outside the schema"


@pytest.mark.parametrize("domain", DOMAINS)
def test_schema_enumerations_cover_observed_gold_values(domain):
    """Every gold value the generator can emit must appear in the schema's enum."""
    schema = OUTPUT_SCHEMAS[domain]
    for task in generate_suite(domain, seed=777, n_tasks=200):
        for key, value in task.gold.items():
            allowed = schema[key]
            if allowed is None or value is None:
                continue
            assert value in allowed, f"{domain}.{key}: gold value {value!r} not in schema enum"


@pytest.mark.parametrize("domain", DOMAINS)
def test_render_prompt_contains_the_input_data(domain):
    """The model must actually receive initial_state, not just the instruction."""
    task = generate_suite(domain, seed=777, n_tasks=1)[0]
    rendered = render_prompt("Do the thing.", task.initial_state, domain)
    for scalar in _scalars(task.initial_state):
        assert scalar in rendered, f"{domain}: {scalar!r} missing from the prompt"


@pytest.mark.parametrize("domain", DOMAINS)
def test_render_prompt_declares_the_output_contract(domain):
    rendered = render_prompt("Do the thing.", {"a": 1}, domain)
    assert "Do the thing." in rendered
    for key in OUTPUT_SCHEMAS[domain]:
        assert f'"{key}"' in rendered


def test_render_prompt_is_deterministic():
    state = {"b": 2, "a": {"z": 1, "y": [3, 4]}}
    first = render_prompt("i", state, "reconciliation")
    second = render_prompt("i", dict(reversed(list(state.items()))), "reconciliation")
    assert first == second


def test_render_prompt_rejects_unknown_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        render_prompt("i", {}, "not_a_domain")


@pytest.mark.parametrize("domain", DOMAINS)
def test_generated_tasks_carry_a_self_contained_prompt(domain):
    """A published task must be answerable from task.prompt alone."""
    for task in generate_suite(domain, seed=777, n_tasks=5):
        for scalar in _scalars(task.initial_state):
            assert scalar in task.prompt, f"{domain}: {scalar!r} missing from task.prompt"
        for key in OUTPUT_SCHEMAS[domain]:
            assert f'"{key}"' in task.prompt


@pytest.mark.parametrize("domain", DOMAINS)
def test_gold_answer_satisfies_the_declared_schema(domain):
    """A response echoing gold must parse and pass the verifier."""
    from lossbench.eval.harness import domain_verifier, parse_outcome

    for task in generate_suite(domain, seed=777, n_tasks=20):
        outcome = parse_outcome(json.dumps(task.gold))
        assert outcome is not None
        assert domain_verifier(task, outcome)


@pytest.mark.parametrize("domain", ["payment_repair", "settlement"])
def test_prompt_states_the_decision_rules(domain):
    """A verdict the verifier derives from a rule must have that rule stated.

    Without this the model is scored on guessing an unpublished convention
    (settlement HIGH/CRITICAL => HITL, payment FRAUD_HOLD => EXCEPTION),
    which measures nothing about the model.
    """
    task = generate_suite(domain, seed=777, n_tasks=1)[0]
    assert "RULES:" in task.prompt


def test_settlement_rules_name_the_hitl_condition():
    task = generate_suite("settlement", seed=777, n_tasks=1)[0]
    assert "HITL" in task.prompt
    assert "CRITICAL" in task.prompt
    assert "ON_TIME" in task.prompt


def test_payment_rules_name_the_fraud_hold_verdict():
    task = generate_suite("payment_repair", seed=777, n_tasks=1)[0]
    assert "FRAUD_HOLD" in task.prompt
    assert "2%" in task.prompt


@pytest.mark.parametrize("domain", DOMAINS)
def test_rules_are_sufficient_to_reach_gold(domain):
    """The stated rules must be complete: gold is reachable from the prompt.

    Checked structurally - every verdict value the generator emits must be
    mentioned in the prompt text for that task.
    """
    for task in generate_suite(domain, seed=777, n_tasks=30):
        assert task.gold["verdict"] in task.prompt


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_task_cost_model_ref_resolves(domain):
    """A published task must not point at a cost profile that does not exist."""
    from lossbench.costs.registry import load_cost_profile

    task = generate_suite(domain, seed=777, n_tasks=1)[0]
    profile = load_cost_profile(task.cost_model_ref)
    assert profile.id == task.cost_model_ref
