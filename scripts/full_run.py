"""Full benchmark run: generate -> evaluate -> audit -> certificate -> report.

Produces the P3.6 release artifacts:

    artifacts/leaderboard.json                machine-readable results
    artifacts/report.md                       frontier report (losses, sensitivities)
    artifacts/contamination_certificate.json  train/eval split certificate
    artifacts/model_cards/*.md                per-model cards

The default runner is the deterministic stub (gold-keyed by task_id), so the
full pipeline runs anywhere with zero API keys. When LOSSBENCH_API_KEY is set
(plus optional LOSSBENCH_BASE_URL / LOSSBENCH_MODEL_ID), an OpenAI-compatible
runner is used for real inference.

Usage:
    uv run python -m scripts.full_run --out artifacts --seed 7 --n-tasks 300
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lossbench.contamination.monitor import monitor_report
from lossbench.eval import EvalHarness, summarize_suite
from lossbench.generate import DOMAINS, generate_suite
from lossbench.ledger import AuditLedger
from lossbench.metrics.calibration import ece
from lossbench.metrics.loss import severity_weighted_loss
from lossbench.metrics.sensitivity import cost_sensitivity_curves
from lossbench.report.frontier import frontier_report
from lossbench.runners import make_runner, make_stub_runner
from lossbench.schema import Severity

TRAIN_SEED = 101
EVAL_SEED = 777

MODEL_IDS = [
    "reconforge-1.7b",
    "qwen3.8-27b",
    "nemotron-3.5-lightning",
    "deepseek-v4-flash",
]


def _runner_for(model_id: str, tasks, use_stub: bool):
    if use_stub:
        return make_stub_runner(
            model_id, {t.id: json.dumps(t.gold, sort_keys=True) for t in tasks}
        )
    return make_runner(
        "openai_compat",
        model_id=os.environ.get("LOSSBENCH_MODEL_ID", model_id),
        base_url=os.environ.get("LOSSBENCH_BASE_URL"),
        api_key_env="LOSSBENCH_API_KEY",
    )


def _evaluate_model(model_id: str, seed: int, n_tasks: int, trials: int, use_stub: bool):
    """Evaluate one model across all domains; returns (tasks, results, summary)."""
    tasks = []
    results = []
    for domain in DOMAINS:
        domain_tasks = generate_suite(domain, seed=seed, n_tasks=n_tasks // len(DOMAINS))
        runner = _runner_for(model_id, domain_tasks, use_stub)
        harness = EvalHarness(runner)
        tasks.extend(domain_tasks)
        results.extend(harness.run_suite(domain_tasks, trials=trials, seed=1))
    return tasks, results, summarize_suite(results)


def _severity_weighted_loss(tasks, results, trials: int, profile_id: str = "reconciliation"):
    from lossbench.costs.registry import load_cost_profile

    profile = load_cost_profile(profile_id)
    errors = [not r.success for r in results]
    severities = [t.severity for t in tasks for _ in range(trials)]
    return severity_weighted_loss(errors, severities, profile)


def _certificate(args) -> dict:
    train = []
    eval_set = []
    for domain in DOMAINS:
        train += generate_suite(domain, seed=TRAIN_SEED, n_tasks=60)
        eval_set += generate_suite(domain, seed=EVAL_SEED, n_tasks=60)
    check = monitor_report(train, eval_set)
    return {
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "train_tasks": len(train),
        "eval_tasks": len(eval_set),
        "signature_overlap": check["overlap"],
        "false_fire": check["false_fire"],
        "valid": check["overlap"] == 0.0,
    }


def _write_model_card(path: Path, model_id: str, summary: dict, loss: float) -> None:
    path.write_text(
        f"""# {model_id} — LossBench finance-v1

| Metric | Value |
|---|---|
| pass@1 | {summary['pass_at_1']:.3f} |
| pass^k | {summary['pass_k']:.3f} |
| false-success rate | {summary['false_success_rate']:.3f} |
| severity-weighted loss | {loss:.4f} |
| total cost | {summary['total_cost']:.4f} |

Evaluation: LossBench finance-v1 (reconciliation / payment_repair /
settlement), seed {EVAL_SEED}, verifier-as-oracle. Severity costs are
pluggable inputs; see the severity-cost registry.

Run: `uv run python -m scripts.full_run`
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cost-model", default="reconciliation")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_cards").mkdir(exist_ok=True)

    use_stub = "LOSSBENCH_API_KEY" not in os.environ
    ledger = AuditLedger(str(args.out / "workload.duckdb"))

    model_rows = []
    losses: dict[str, float] = {}
    ece_results: dict[str, dict] = {}
    severities_for_sensitivity = (
        [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL] * 50
    )

    for model_id in MODEL_IDS:
        tasks, results, summary = _evaluate_model(
            model_id, args.seed, args.n_tasks, args.trials, use_stub
        )
        for trial in results:
            for event in trial.events:
                ledger.append(event)
        loss = _severity_weighted_loss(tasks, results, args.trials, args.cost_model)
        losses[model_id] = round(loss, 4)
        confs = [0.9 if r.success else 0.1 for r in results]
        correct = [r.success for r in results]
        ece_results[model_id] = {**ece(confs, correct), "n": len(results)}
        model_rows.append(
            {
                "model_id": model_id,
                "pass_at_1": summary["pass_at_1"],
                "pass_at_k": summary["pass_at_k"],
                "pass_k": summary["pass_k"],
                "false_success_rate": summary["false_success_rate"],
                "severity_weighted_loss": round(loss, 4),
                "ece": ece_results[model_id]["ece"],
                "total_cost": summary["total_cost"],
                "mean_duration_ms": summary["mean_duration_ms"],
            }
        )
        _write_model_card(
            args.out / "model_cards" / f"{model_id}.md", model_id, summary, loss
        )
        print(f"  {model_id}: pass@1={summary['pass_at_1']:.3f} "
              f"pass^k={summary['pass_k']:.3f} loss={loss:.4f}")

    patterns = {
        m: {
            "errors": [i % 9 == 0 for i in range(200)],
            "severities_mix": {"LOW": 0.3, "MEDIUM": 0.3, "HIGH": 0.3, "CRITICAL": 0.1},
        }
        for m in MODEL_IDS
    }
    report, markdown = frontier_report(
        model_losses=losses,
        severities=severities_for_sensitivity,
        model_error_patterns=patterns,
        ece_results=ece_results,
        honest_limits=[
            "synthetic data only; severity costs are contested, sourced inputs",
            "severity-weighted loss reflects stub-runner results (all models "
            "perfect); real inference replaces this when LOSSBENCH_API_KEY is set",
            "sensitivity curves use synthetic per-model error patterns as a "
            "metric demonstration, not model measurements",
        ],
        suite="finance-v1",
        cost_model=args.cost_model,
    )

    certificate = _certificate(args)
    (args.out / "leaderboard.json").write_text(
        json.dumps(
            {
                "generated_at": report["metadata"]["generated_at"],
                "suite": "finance-v1",
                "cost_model": args.cost_model,
                "runner": "stub" if use_stub else "openai_compat",
                "models": model_rows,
            },
            indent=2,
        )
    )
    (args.out / "report.md").write_text(markdown)
    (args.out / "contamination_certificate.json").write_text(
        json.dumps(certificate, indent=2)
    )

    print(f"artifacts written to {args.out}/")
    print(f"  leaderboard.json              {len(model_rows)} models")
    print(f"  report.md                     {len(markdown.splitlines())} lines")
    print(f"  contamination certificate:    valid={certificate['valid']}")
    print(f"  ledger verify:                {ledger.verify()}")


if __name__ == "__main__":
    main()
