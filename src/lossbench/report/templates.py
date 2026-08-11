"""Schema-agnostic rendering primitives shared by markdown and HTML reports.

The report package keeps assembly (generator.py) separate from rendering
(templates.py): both renderers consume the same neutral Section list, which
guarantees identical content and ordering across output formats.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

CSS = (
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
    "Helvetica,Arial,sans-serif;max-width:960px;margin:2rem auto;"
    "padding:0 1rem;color:#1a1a1a;line-height:1.5}"
    "h1{border-bottom:2px solid #333;padding-bottom:.3rem}"
    "h2{margin-top:1.8rem}"
    "section{margin:1.5rem 0}"
    "table{border-collapse:collapse;margin:.5rem 0;width:100%}"
    "th,td{border:1px solid #ccc;padding:.4rem .7rem;text-align:left}"
    "th{background:#f4f4f4}"
    "code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}"
    "ul,ol{margin:.3rem 0}"
)


@dataclass(frozen=True)
class Section:
    """A named report section holding uniform rows for either renderer.

    kind is one of "table" (rows are cell tuples matching header), "bullets"
    (rows are (label, value) pairs), or "numbered" (rows are one-tuples).
    """

    id: str
    heading: str
    kind: str
    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()


def format_number(value: float) -> str:
    """Render any real number with exactly 4 decimals, deterministically."""
    return f"{float(value):.4f}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render an aligned-pipe markdown table from a header and string rows."""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(title: str, sections: Sequence[Section]) -> str:
    """Render the full markdown document from neutral sections."""
    blocks = [f"# {title}"]
    for section in sections:
        blocks.append(f"## {section.heading}")
        if section.kind == "table":
            blocks.append(md_table(section.header, section.rows))
        elif section.kind == "bullets":
            blocks.extend(f"- {label}: {value}" for label, value in section.rows)
        else:
            blocks.extend(f"{i}. {row[0]}" for i, row in enumerate(section.rows, start=1))
    return "\n\n".join(blocks) + "\n"


def _render_html_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["<table>", "<thead><tr>"]
    lines.append("".join(f"<th>{escape(cell)}</th>" for cell in header))
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    lines.append("</tbody></table>")
    return lines


def render_html(title: str, sections: Sequence[Section]) -> str:
    """Render a self-contained HTML document (inline CSS, no external refs)."""
    body = [f"<h1>{escape(title)}</h1>"]
    for section in sections:
        body.append(f'<section id="{escape(section.id)}">')
        body.append(f"<h2>{escape(section.heading)}</h2>")
        if section.kind == "table":
            body.extend(_render_html_table(section.header, section.rows))
        elif section.kind == "bullets":
            body.append("<ul>")
            for label, value in section.rows:
                body.append(f"<li><code>{escape(label)}</code>: {escape(value)}</li>")
            body.append("</ul>")
        else:
            body.append("<ol>")
            for (item,) in section.rows:
                body.append(f"<li>{escape(item)}</li>")
            body.append("</ol>")
        body.append("</section>")
    head = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{escape(title)}</title>\n<style>{CSS}</style>\n"
        "</head>\n<body>\n"
    )
    return head + "\n".join(body) + "\n</body>\n</html>\n"
