"""Strict response parsing: a prose answer is a parse miss, not a verdict."""

from __future__ import annotations

from lossbench.eval.harness import TrialResult, parse_outcome, summarize_suite


def test_parses_bare_json():
    assert parse_outcome('{"verdict": "MATCH", "exception_type": null}') == {
        "verdict": "MATCH",
        "exception_type": None,
    }


def test_parses_fenced_block():
    text = 'Here you go:\n```json\n{"verdict": "EXCEPTION"}\n```\nHope that helps.'
    assert parse_outcome(text) == {"verdict": "EXCEPTION"}


def test_parses_unlabelled_fence():
    assert parse_outcome('```\n{"verdict": "MATCH"}\n```') == {"verdict": "MATCH"}


def test_strips_reasoning_preamble():
    text = '<think>The amounts differ, so this is an exception.</think>{"verdict": "EXCEPTION"}'
    assert parse_outcome(text) == {"verdict": "EXCEPTION"}


def test_recovers_object_embedded_in_prose():
    text = 'The answer is {"verdict": "MATCH", "exception_type": null} as shown.'
    assert parse_outcome(text) == {"verdict": "MATCH", "exception_type": None}


def test_prose_is_a_parse_miss_not_a_verdict():
    """The old fallback turned a whole paragraph into a verdict string."""
    assert parse_outcome("The pair looks like a match to me.") is None


def test_non_object_json_is_a_parse_miss():
    assert parse_outcome("[1, 2, 3]") is None
    assert parse_outcome('"MATCH"') is None
    assert parse_outcome("") is None


def _result(task_id: str, *, success: bool, parse_ok: bool) -> TrialResult:
    return TrialResult(
        task_id=task_id,
        model_id="m",
        success=success,
        events=[],
        duration_ms=1.0,
        cost=0.0,
        parse_ok=parse_ok,
    )


def test_summarize_reports_parse_rate():
    results = [
        _result("a", success=True, parse_ok=True),
        _result("b", success=False, parse_ok=False),
    ]
    assert summarize_suite(results)["parse_rate"] == 0.5


def test_summarize_empty_reports_zero_parse_rate():
    assert summarize_suite([])["parse_rate"] == 0.0
