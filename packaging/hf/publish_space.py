"""Publish the control-plane demo as a Hugging Face Space.

The Space is a flat directory: `demo.py` goes up as `app.py`, which is the
name the README front matter declares. Nothing here depends on a benchmark
run, so the demo is honest whether or not model results exist.

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

# source name -> name in the Space repo
FILES = {
    "demo.py": "app.py",
    "requirements.txt": "requirements.txt",
    "README.md": "README.md",
}


def build_payload(out: Path) -> list[Path]:
    """Copy the Space files into `out` under the names the Hub expects."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for source_name, target_name in FILES.items():
        source = _SOURCE / source_name
        if not source.exists():
            raise SystemExit(f"missing Space file: {source}")
        target = out / target_name
        shutil.copyfile(source, target)
        written.append(target)

    readme = out / "README.md"
    head = readme.read_text(encoding="utf-8").split("---\n")
    if len(head) < 3:
        raise SystemExit(
            "Space README has no YAML front matter; the Hub cannot resolve an SDK"
        )
    if "app_file: app.py" not in head[1]:
        raise SystemExit("Space README front matter must declare app_file: app.py")
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
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=str(args.out),
        commit_message="control-plane demo: replay a workload under a new policy",
    )
    print(f"published https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
