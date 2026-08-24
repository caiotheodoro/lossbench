"""Publish the control-plane demo as a static Hugging Face Space.

Gradio and Docker Spaces require a paid plan; static Spaces are free. The
demo is precomputed by `static_space.build_data`, which scores every
threshold with ReplayLab at build time, so the published page ships numbers
this repo computed rather than a JavaScript reimplementation of the replay.

The Gradio app in `spaces/demo/demo.py` still runs locally and stays the
reference implementation. Nothing here depends on a benchmark run, so the
demo is honest whether or not model results exist.

    uv run python packaging/hf/publish_space.py --dry-run
    uv run python packaging/hf/publish_space.py --repo caiotheodoro/lossbench-demo

Requires HF_TOKEN in the environment, or a prior `hf auth login`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SOURCE = _REPO_ROOT / "spaces" / "demo"

DEFAULT_REPO = "caiotheodoro/lossbench-demo"

def build_payload(out: Path) -> list[Path]:
    """Precompute the replay and write the static Space into `out`."""
    import sys

    sys.path.insert(0, str(_HERE))
    import static_space

    out.mkdir(parents=True, exist_ok=True)
    written = []

    data = static_space.build_data()
    page = out / "index.html"
    page.write_text(static_space.build_page(data), encoding="utf-8")
    written.append(page)

    readme = out / "README.md"
    shutil.copyfile(_SOURCE / "README.md", readme)
    written.append(readme)

    head = readme.read_text(encoding="utf-8").split("---\n")
    if len(head) < 3:
        raise SystemExit(
            "Space README has no YAML front matter; the Hub cannot resolve an SDK"
        )
    if "sdk: static" not in head[1]:
        raise SystemExit("Space README front matter must declare sdk: static")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=Path("/tmp/lossbench-space"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    written = build_payload(args.out)

    print(f"payload built in {args.out}")
    for path in written:
        print(f"  {path.name}  {path.stat().st_size:,} bytes")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=str(args.out),
        commit_message=(
            "static control-plane demo: replay precomputed by ReplayLab "
            "across all 101 thresholds"
        ),
    )
    print(f"published https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
