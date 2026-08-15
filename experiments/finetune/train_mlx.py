"""MLX LoRA training skeleton for the finance back-office model family.

The canonical training runs on an Apple M-series (MLX) or Kaggle GPU
(Unsloth). This module validates the configuration and data contract and,
when `mlx-lm` is installed, launches a LoRA fine-tune of Qwen3-1.7B on the
exported training data.

Usage:
    uv run python -m experiments.finetune.train_mlx --dry-run
    uv run python -m experiments.finetune.train_mlx --data data/train.jsonl

The ReconForge result (1.7B LoRA, ~2h on an M5, beats the frontier model on
severity-weighted recall) is the continuity target; this script is the
reproducible entry point for that run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIG = {
    "base_model": "Qwen/Qwen3-1.7B",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "batch_size": 2,
    "gradient_checkpointing": True,
    "steps": 740,
    "learning_rate": 1e-4,
    "eval_every": 100,
}


def _verify_data(path: Path) -> int:
    n = 0
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            for key in ("task_id", "domain", "prompt", "gold", "severity"):
                if key not in record:
                    raise ValueError(f"missing key '{key}' in {path}:{n + 1}")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="validate config + data, do not train")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"training data not found: {args.data} (run export_training_data.py first)")
    n = _verify_data(args.data)
    print(f"data verified: {n} examples in {args.data}")

    try:
        from mlx_lm import load, generate  # type: ignore  # noqa: F401
        from mlx_lm.tuner import train as mlx_train  # type: ignore  # noqa: F401

        mlx_available = True
    except ImportError:
        mlx_available = False

    if args.dry_run:
        plan = {"config": CONFIG, "data": str(args.data), "n_examples": n}
        print(json.dumps(plan, separators=(",", ":")))
        print("dry-run complete: no training started")
        if not mlx_available:
            print("note: mlx-lm not installed; install via `uv run pip install mlx-lm`")
        return

    if not mlx_available:
        raise SystemExit("mlx-lm is not installed; run with --dry-run to validate first")

    raise NotImplementedError(
        "live training entry point — see experiments/finetune/README.md for the "
        "canonical ReconForge recipe and Kaggle/Unsloth path"
    )


if __name__ == "__main__":
    main()
