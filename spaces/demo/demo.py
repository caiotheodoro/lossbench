"""Interactive control-plane demo Space: re-run last month's workload.

Wraps the ReplayLab counterfactual in a Gradio UI. All replay math lives in
ReplayLab; this module only generates the workload and renders results.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import Any

import gradio as gr

from lossbench.costs.registry import list_cost_profiles, load_cost_profile
from lossbench.eval.harness import EvalHarness
from lossbench.generate import generate_suite
from lossbench.ledger.store import AuditLedger
from lossbench.replay.simulator import ReplayLab, ReplayOutcome
from lossbench.runners import make_stub_runner
from lossbench.schema import PolicyBundle


def _gold_text(task: Any) -> str:
    """Gold answer plus a confidence spread across the workload.

    The recorded workload is a stub replay, so confidence is derived from
    task difficulty rather than measured. Without a spread every event
    carries the same probability and no threshold change can flip a
    decision, which makes the replay slider inert.
    """
    return json.dumps(
        {**task.gold, "confidence": round(1.0 - float(task.difficulty), 3)},
        sort_keys=True,
    )


def build_workload(seed: int = 7, n_tasks: int = 200) -> tuple[list[Any], AuditLedger]:
    """Generate a reconciliation suite, run it through the EvalHarness with a
    stub runner (gold-keyed, task_id lookup), append every event to an
    AuditLedger, and return (events, ledger). Deterministic per seed."""
    tasks = generate_suite("reconciliation", seed=seed, n_tasks=n_tasks)
    responses = {task.id: _gold_text(task) for task in tasks}
    harness = EvalHarness(runner=make_stub_runner("demo-stub", responses))
    ledger = AuditLedger()
    events: list[Any] = []
    for result in harness.run_suite(tasks, trials=1, seed=seed):
        for event in result.events:
            ledger.append(event)
            events.append(event)
    return events, ledger


@functools.lru_cache(maxsize=1)
def _workload(seed: int, n_tasks: int) -> tuple[Any, ...]:
    events, _ = build_workload(seed, n_tasks)
    return tuple(events)


def _lab(cost_model: str) -> ReplayLab:
    return ReplayLab(load_cost_profile(cost_model))


def _policy(cost_model: str, threshold: float) -> PolicyBundle:
    return PolicyBundle(id="demo", cost_model_id=cost_model, escalation_threshold=threshold)


def _render(outcome: ReplayOutcome, n_events: int) -> str:
    return (
        "## Re-run last month\n"
        f"- events replayed: {n_events}\n"
        f"- cases whose decision changed: {len(outcome.per_case_diff)}\n"
        f"- before: loss {outcome.before_loss}, review load {outcome.before_review_load}\n"
        f"- after: loss {outcome.after_loss}, review load {outcome.after_review_load}"
    )


def simulate_ui(
    events: Sequence[Any],
    policy_threshold: float,
    new_threshold: float,
    cost_model: str = "reconciliation",
) -> dict[str, Any]:
    """Return the documented outcome dict for one threshold replay.

    Keys: before_loss, after_loss, before_review_load, after_review_load,
    n_events, n_cases_changed, markdown. markdown is a short render of the
    outcome (header + per-case diff count + before/after). Delegates to
    ReplayLab; never re-implements replay math.
    """
    outcome = _lab(cost_model).simulate(
        events, _policy(cost_model, policy_threshold), new_threshold
    )
    return {
        "before_loss": outcome.before_loss,
        "after_loss": outcome.after_loss,
        "before_review_load": outcome.before_review_load,
        "after_review_load": outcome.after_review_load,
        "n_events": len(events),
        "n_cases_changed": len(outcome.per_case_diff),
        "markdown": _render(outcome, len(events)),
    }


def _diff_rows(outcome: ReplayOutcome) -> list[list[str]]:
    rows: list[list[str]] = []
    for diff in outcome.per_case_diff:
        expected = diff.get("expected_loss")
        rows.append(
            [
                str(diff["event_id"]),
                str(diff["before"]),
                str(diff["after"]),
                "" if expected is None else f"{expected:.4f}",
            ]
        )
    return rows


def _on_replay(
    policy_threshold: float,
    new_threshold: float,
    cost_model: str,
    n_tasks: int = 200,
) -> tuple[str, list[list[str]]]:
    """Replay once and render both outputs from the single outcome.

    Simulating twice per click doubled the work for identical results.
    """
    events = _workload(7, int(n_tasks))
    outcome = _lab(cost_model).simulate(
        events, _policy(cost_model, policy_threshold), new_threshold
    )
    return _render(outcome, len(events)), _diff_rows(outcome)


def demo() -> gr.Blocks:
    """Build the Gradio Blocks for the control-plane demo Space."""
    _workload(7, 200)
    profiles = list_cost_profiles()
    with gr.Blocks(title="LossBench — re-run last month") as blocks:
        gr.Markdown(
            "## LossBench control plane\n"
            "Re-run last month's reconciliation workload under a different "
            "escalation threshold — no model calls, fully deterministic.\n\n"
            "This is a recorded stub workload, so the confidences are derived "
            "from task difficulty rather than measured from a model."
        )
        with gr.Row():
            policy_threshold = gr.Slider(
                0.0, 1.0, value=0.5, step=0.01, label="Last month's policy threshold"
            )
            new_threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="New threshold")
            cost_model = gr.Dropdown(profiles, value="reconciliation", label="Cost model")
        workload_size = gr.Slider(
            20, 400, value=200, step=20, label="Workload size (tasks)"
        )
        replay = gr.Button("Replay")
        outcome_markdown = gr.Markdown()
        per_case_frame = gr.Dataframe(
            headers=["event_id", "before", "after", "expected_loss"],
            interactive=False,
        )
        replay.click(
            _on_replay,
            inputs=[policy_threshold, new_threshold, cost_model, workload_size],
            outputs=[outcome_markdown, per_case_frame],
        )
    return blocks


if __name__ == "__main__":
    demo().launch()
