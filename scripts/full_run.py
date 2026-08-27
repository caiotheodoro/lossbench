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
from lossbench.report.frontier import frontier_report
from lossbench.runners import make_runner, make_stub_runner
from lossbench.runners.baselines import BASELINE_MODELS
from lossbench.runners.budget import BudgetedRunner, BudgetExceeded, BudgetTracker
from lossbench.runners.retry import RetryingRunner
from lossbench.schema import Severity

TRAIN_SEED = 101
EVAL_SEED = 777

MODEL_IDS = [
    "reconforge-1.7b",
    "qwen3.8-27b",
    "nemotron-3.5-lightning",
    "deepseek-v4-flash",
]


def _runner_for(model_id: str, tasks, use_stub: bool, tracker=None, base_url=None):
    """Build the runner for one model.

    The stub is gold-keyed by task id, so it bypasses the prompt entirely
    and every model scores perfectly; it is a pipeline smoke path, never a
    result. Real runners are wrapped in BudgetedRunner so one ceiling
    covers the whole run rather than one ceiling per model.
    """
    if use_stub:
        return make_stub_runner(
            model_id, {t.id: json.dumps(t.gold, sort_keys=True) for t in tasks}
        )
    pricing = BASELINE_MODELS.get(model_id, {})
    runner = make_runner(
        "openai_compat",
        model_id=model_id,
        base_url=base_url or os.environ.get("LOSSBENCH_BASE_URL"),
        api_key_env="LOSSBENCH_API_KEY",
        cost_per_1k_in=pricing.get("cost_per_1k_in", 0.0),
        cost_per_1k_out=pricing.get("cost_per_1k_out", 0.0),
    )
    runner = RetryingRunner(runner)
    return runner if tracker is None else BudgetedRunner(runner, tracker)


def _evaluate_model(
    model_id: str,
    seed: int,
    n_tasks: int,
    trials: int,
    use_stub: bool,
    max_steps: int = 8,
    tracker=None,
    base_url=None,
):
    """Evaluate one model across all domains; returns (tasks, results, summary)."""
    tasks = []
    results = []
    for domain in DOMAINS:
        domain_tasks = generate_suite(domain, seed=seed, n_tasks=n_tasks // len(DOMAINS))
        runner = _runner_for(model_id, domain_tasks, use_stub, tracker, base_url)
        harness = EvalHarness(runner, max_steps=max_steps)
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


def _honest_limits(use_stub: bool) -> list[str]:
    """Limits that match the run that actually happened."""
    limits = [
        "synthetic data only; severity costs are contested, sourced inputs",
        "sensitivity curves use synthetic per-model error patterns as a "
        "metric demonstration, not model measurements",
    ]
    if use_stub:
        limits.insert(
            1,
            "STUB RUN: the runner is gold-keyed by task id, so every model "
            "scores perfectly by construction. These are not model results. "
            "Set LOSSBENCH_API_KEY for real inference.",
        )
    else:
        limits.insert(1, "single run, one seed; parse failures count as misses")
    return limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cost-model", default="reconciliation")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model ids to evaluate (default: the placeholder set)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="corrective retries per task; keep low for paid runs",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.0,
        help="hard USD ceiling for the whole run; 0 means unlimited",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    args = parser.parse_args()
    model_ids = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(MODEL_IDS)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_cards").mkdir(exist_ok=True)

    use_stub = "LOSSBENCH_API_KEY" not in os.environ
    tracker = None if use_stub else BudgetTracker(args.max_cost)
    ledger = AuditLedger(str(args.out / "workload.duckdb"))

    model_rows = []
    losses: dict[str, float] = {}
    ece_results: dict[str, dict] = {}
    budget_abort_msg: str | None = None
    severities_for_sensitivity = (
        [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL] * 50
    )

    for model_id in model_ids:
        try:
            tasks, results, summary = _evaluate_model(
                model_id,
                args.seed,
                args.n_tasks,
                args.trials,
                use_stub,
                max_steps=args.max_steps,
                tracker=tracker,
                base_url=args.base_url,
            )
        except BudgetExceeded as exc:
            print(f"  ABORT before finishing {model_id}: {exc}")
            budget_abort_msg = (
                f"Run aborted by BudgetExceeded before {model_id} "
                f"({len(model_rows)} of {len(model_ids)} models completed): {exc}"
            )
            break
        for trial in results:
            for event in trial.events:
                ledger.append(event)
        loss = _severity_weighted_loss(tasks, results, args.trials, args.cost_model)
        losses[model_id] = round(loss, 4)
        # Confidence comes from the model's own answer (harness._confidence).
        # Deriving it from correctness here would make ECE a constant.
        confs = [
            trial.events[-1].calibrated_probability if trial.events else 0.5
            for trial in results
        ]
        correct = [r.success for r in results]
        ece_results[model_id] = {**ece(confs, correct), "n": len(results)}
        model_rows.append(
            {
                "model_id": model_id,
                "pass_at_1": summary["pass_at_1"],
                "pass_at_k": summary["pass_at_k"],
                "pass_k": summary["pass_k"],
                "false_success_rate": summary["false_success_rate"],
                "parse_rate": summary["parse_rate"],
                "error_rate": summary["error_rate"],
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
              f"pass^k={summary['pass_k']:.3f} parse={summary['parse_rate']:.3f} "
              f"err={summary['error_rate']:.3f} loss={loss:.4f}")

    patterns = {
        m: {
            "errors": [i % 9 == 0 for i in range(200)],
            "severities_mix": {"LOW": 0.3, "MEDIUM": 0.3, "HIGH": 0.3, "CRITICAL": 0.1},
        }
        for m in model_ids
    }
    report, markdown = frontier_report(
        model_losses=losses,
        severities=severities_for_sensitivity,
        model_error_patterns=patterns,
        ece_results=ece_results,
        honest_limits=_honest_limits(use_stub),
        suite="finance-v1",
        cost_model=args.cost_model,
    )

    certificate = _certificate(args)
    # partial is true for a stub run (always) OR a real run that a BudgetExceeded
    # abort cut short (model_rows has fewer entries than model_ids) -- either
    # way this is not a complete, final result and must not render as one.
    is_partial = use_stub or budget_abort_msg is not None
    leaderboard_doc: dict = {
        "generated_at": report["metadata"]["generated_at"],
        "suite": "finance-v1",
        "cost_model": args.cost_model,
        "runner": "stub" if use_stub else "openai_compat",
        "partial": is_partial,
        "models": model_rows,
    }
    if use_stub:
        # The stub is gold-keyed, so every model scores perfectly — this file
        # is a pipeline smoke artifact, never a result. The leaderboard Space
        # renders this banner as a loud warning (see issue #2); the publish
        # gate stays closed until a real run replaces it (issue #23).
        leaderboard_doc["banner"] = "STUB PIPELINE SMOKE OUTPUT"
        leaderboard_doc["partial_note"] = "Stub pipeline smoke output, not a real run."
    elif budget_abort_msg is not None:
        leaderboard_doc["banner"] = "PARTIAL RUN — BUDGET-ABORTED"
        leaderboard_doc["partial_note"] = budget_abort_msg
    (args.out / "leaderboard.json").write_text(json.dumps(leaderboard_doc, indent=2))
    (args.out / "report.md").write_text(markdown)
    (args.out / "contamination_certificate.json").write_text(
        json.dumps(certificate, indent=2)
    )

    print(f"artifacts written to {args.out}/")
    print(f"  leaderboard.json              {len(model_rows)} models")
    print(f"  report.md                     {len(markdown.splitlines())} lines")
    print(f"  contamination certificate:    valid={certificate['valid']}")
    print(f"  ledger verify:                {ledger.verify()}")
    if tracker is not None:
        print(f"  spend:                        ${tracker.spent:.4f}")


if __name__ == "__main__":
    main()
