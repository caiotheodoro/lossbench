"""LossBench P3.1: thin Gradio UI for the leaderboard Space.

All logic lives in leaderboard.py; this module only wires components.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from leaderboard import (
    _format_honest_limits,
    _load_extras,
    crossover_summary,
    load_leaderboard,
    render_table,
)

_SAMPLE_PATH = Path(__file__).resolve().with_name("sample_leaderboard.json")


def demo() -> gr.Blocks:
    """Build the leaderboard Space UI; returns a gr.Blocks without launching.

    Uploads a leaderboard JSON (default: the embedded sample) and renders the
    loss table, cost-sensitivity crossover summary, and honest limits.
    """

    def refresh(path: str) -> tuple[str, str, str]:
        sensitivities, limits = _load_extras(path)
        return (
            render_table(load_leaderboard(path)),
            crossover_summary(sensitivities),
            _format_honest_limits(limits),
        )

    table_md, crossover_md, limits_md = refresh(str(_SAMPLE_PATH))
    with gr.Blocks(title="LossBench Leaderboard") as blocks:
        gr.Markdown(
            "# LossBench Leaderboard\n"
            "Severity-weighted loss for agentic back-office models across cost "
            "regimes. Upload a leaderboard JSON (frontier report artifact) to "
            "browse a different run."
        )
        file_input = gr.File(
            label="Leaderboard JSON", file_types=[".json"], value=str(_SAMPLE_PATH)
        )
        gr.Markdown("## Leaderboard")
        table_out = gr.Markdown(table_md)
        gr.Markdown("## Cost-Sensitivity Crossovers")
        crossover_out = gr.Markdown(crossover_md)
        gr.Markdown("## Honest Limits")
        limits_out = gr.Markdown(limits_md)
        file_input.change(
            refresh, inputs=file_input, outputs=[table_out, crossover_out, limits_out]
        )
    return blocks


if __name__ == "__main__":
    demo().launch()
