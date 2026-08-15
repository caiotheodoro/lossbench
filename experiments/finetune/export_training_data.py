"""Export verifier-validated training data from the benchmark generators.

Contamination discipline (from the design spec): training split from seed 101,
evaluation split from seed 777, zero signature overlap enforced and reported
as a certificate alongside the export.

Usage:
    uv run python -m experiments.finetune.export_training_data \
        --out data/train.jsonl --eval-out data/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lossbench.contamination.monitor import monitor_report, task_signature
from lossbench.generate import DOMAINS, generate_suite

TRAIN_SEED = 101
EVAL_SEED = 777
TRAIN_N = 1500
EVAL_N = 400
DOMAIN_MIX = {"reconciliation": 0.6, "payment_repair": 0.25, "settlement": 0.15}


def _suite_for(domain: str, seed: int, n: int):
    if domain == "reconciliation":
        return generate_suite(domain, seed=seed, n_tasks=n)
    return generate_suite(domain, seed=seed, n_tasks=n)


def _export(tasks, path: Path) -> int:
    with path.open("w") as fh:
        for task in tasks:
            record = {
                "task_id": task.id,
                "domain": task.domain,
                "prompt": task.prompt,
                "initial_state": task.initial_state,
                "severity": task.severity.value,
                "gold": task.gold,
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--train-n", type=int, default=TRAIN_N)
    parser.add_argument("--eval-n", type=int, default=EVAL_N)
    args = parser.parse_args()

    train: list = []
    eval_set: list = []
    for domain in DOMAINS:
        share = DOMAIN_MIX[domain]
        train += _suite_for(domain, TRAIN_SEED, int(args.train_n * share))
        eval_set += _suite_for(domain, EVAL_SEED, int(args.eval_n * share))

    n_train = _export(train, args.out)
    n_eval = _export(eval_set, args.eval_out)

    cert = monitor_report(train, eval_set)
    print(
        json.dumps(
            {
                "train_tasks": n_train,
                "eval_tasks": n_eval,
                "train_signatures": len({task_signature(t) for t in train}),
                "eval_signatures": len({task_signature(t) for t in eval_set}),
                "contamination": cert,
                "note": "signature overlap must be 0.0 for a valid split",
            },
            indent=2,
        )
    )
    if cert["overlap"] != 0.0:
        raise SystemExit("contamination detected: train/eval signature overlap != 0")


if __name__ == "__main__":
    main()
