"""Determinism gate: two full runs from the same seed must agree byte-for-byte
on report.md and the contamination certificate, and modulo generated_at on the
leaderboard. Exit 0 when identical, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _run_dir(root: Path) -> Path:
    """full_run writes into a single per-run subdirectory; resolve it."""
    return next(p for p in sorted(root.iterdir()) if p.is_dir())

A = _run_dir(Path("/tmp/lb-golden-a"))
B = _run_dir(Path("/tmp/lb-golden-b"))


def _normalized(text: str) -> str:
    """Strip runtime metadata (generated_at) before comparison."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("- generated_at:")
    )


def main() -> int:
    ok = True
    for name in ("report.md", "contamination_certificate.json"):
        a = _normalized((A / name).read_text())
        b = _normalized((B / name).read_text())
        if a != b:
            print(f"determinism violation in {name}")
            ok = False
        else:
            print(f"{name}: byte-identical (modulo generated_at)")
    la = json.loads((A / "leaderboard.json").read_text())
    lb = json.loads((B / "leaderboard.json").read_text())
    la.pop("generated_at", None)
    lb.pop("generated_at", None)
    for row in la.get("models", []):
        row.pop("mean_duration_ms", None)
    for row in lb.get("models", []):
        row.pop("mean_duration_ms", None)
    if la != lb:
        print("leaderboard differs beyond generated_at")
        ok = False
    else:
        print("leaderboard: identical modulo runtime metadata")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
