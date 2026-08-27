"""Publish the LossBench suite to the Hugging Face Hub.

The only code in this repo that writes to the Hub. Everything it uploads is
regenerated from seeds at publish time, so the dataset is a pure function of
(generator version, seed, n_tasks) rather than a blob that has to be trusted.

    uv run python -m packaging.hf.publish --dry-run
    uv run python -m packaging.hf.publish --repo caiotheodoro/lossbench-finance-v1

Requires HF_TOKEN in the environment, or a prior `hf auth login`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from lossbench.contamination.monitor import monitor_report
from lossbench.costs.registry import load_cost_profile
from lossbench.generate import DOMAINS, generate_suite
from lossbench.schema import Severity

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

TRAIN_SEED = 101
EVAL_SEED = 777
DEFAULT_REPO = "caiotheodoro/lossbench-finance-v1"
LICENSE = "cc-by-4.0"
COST_MODELS = ["flat", "reconciliation", "principal_risk", "review_heavy"]
SEVERITY_BANDS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _load_exporter():
    """Load the sibling exporter by path; packaging/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "lossbench_hf_exporter", _HERE / "exporter.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _suite(seed: int, per_domain: int):
    tasks = []
    for domain in DOMAINS:
        tasks.extend(generate_suite(domain, seed=seed, n_tasks=per_domain))
    return tasks


def _results_table(leaderboard: Path) -> tuple[str, str]:
    """Render the results table and its provenance line from a real run.

    Returns the "no results" placeholder when the run was a stub, because a
    stub run is gold-keyed by task id and every model scores perfectly by
    construction. Those numbers are not model results and must never ship.
    """
    if not leaderboard.exists():
        return ("No model results published for this revision yet.", "")
    data = json.loads(leaderboard.read_text(encoding="utf-8"))
    if data.get("runner") != "openai_compat" or not data.get("models"):
        return (
            "No model results published for this revision yet. The only local "
            "run used the gold-keyed stub runner, which scores every model "
            "perfectly by construction.",
            "",
        )
    rows = sorted(data["models"], key=lambda r: r["severity_weighted_loss"])
    # Only render columns every row actually carries. A run that was stopped
    # before the summary was written has no ECE or parse rate, and printing
    # "nan" would read as a measured value.
    # False-success is only a real measurement for CLAIM_THEN_VERIFY sources
    # (dsh/langgraph adapters). Harness-scored rows carry false_success_applicable
    # =False and a null rate; render neither as a bare 0.0 that reads as a clean
    # bill of health (issue #10). Read the flag full_run.py already computed and
    # threaded into each row -- don't re-derive it by sniffing the rate's type,
    # which can silently disagree with the flag if a future source's rate and
    # applicability signals ever diverge.
    fs_applicable = all(r.get("false_success_applicable") is True for r in rows)
    optional = [("ece", "ECE"), ("parse_rate", "Parse"), ("error_rate", "Errors")]
    extra = [(key, label) for key, label in optional if all(key in r for r in rows)]
    fs_col = " | False-success" if fs_applicable else ""
    fs_sep = "|---:" if fs_applicable else ""
    header = (
        "| Model | Expected loss | pass@1 | pass^k"
        + fs_col
        + "".join(f" | {label}" for _, label in extra)
        + " |\n|---|---:|---:|---:"
        + fs_sep
        + "".join("|---:" for _ in extra)
        + "|\n"
    )
    body = "".join(
        f"| `{r['model_id']}` | {r['severity_weighted_loss']:.4f} "
        f"| {r['pass_at_1']:.3f} | {r['pass_k']:.3f}"
        + (f" | {r['false_success_rate']:.3f}" if fs_applicable else "")
        + "".join(f" | {r[key]:.3f}" for key, _ in extra)
        + " |\n"
        for r in rows
    )
    shape = ""
    if data.get("n_tasks"):
        shape = (
            f" {data['n_tasks']} tasks x {data.get('trials', 1)} trials, "
            f"single-shot ({data.get('max_steps', 1)} step, no corrective retry)"
            + (f", via {data['endpoint']}" if data.get("endpoint") else "")
            + "."
        )
    partial = f"\n\n**{data['partial_note']}**" if data.get("partial") else ""
    note = (
        f"\nRanked by expected loss under the `{data['cost_model']}` cost model, "
        f"lower is better. Seed {EVAL_SEED}, generated {data['generated_at']}, "
        f"runner `{data['runner']}`.{shape}{partial}\n\n"
        "- **Expected loss** charges each unreviewed error the severity cost "
        "`K` of the task it got wrong, so a HIGH miss outweighs a pile of LOW "
        "ones. It is not accuracy.\n"
        "- Token cost is deliberately omitted: the repo prices runs from a "
        "placeholder rate table, not from what the gateway actually billed.\n"
    )
    if not fs_applicable:
        note += (
            "- **False-success** is not reported: these runs use the "
            "self-verifying eval harness, where every decision is checked "
            "against gold by construction, so the "
            "\"claimed done, nothing verified it\" rate is a structural 0.0 "
            "rather than a measurement (issue #10). It is a real metric for "
            "the dsh / langgraph adapters.\n"
        )
    shown = {key for key, _ in extra}
    if "ece" in shown:
        note += (
            "- **ECE** is measured against the `confidence` each model reports "
            "in its own answer, which is never graded against the answer key.\n"
        )
    if shown & {"parse_rate", "error_rate"}:
        note += (
            "- **Parse** failures and **errors** both count as misses. Errors "
            "are gateway failures, not model mistakes, and are listed "
            "separately so they cannot be read as one.\n"
        )
    divergence = _divergence_note(rows)
    if divergence:
        note += "\n" + divergence + "\n"
    return header + body + note, data["generated_at"]


def _divergence_note(rows: list[dict]) -> str:
    """Call out any pair where accuracy and expected loss disagree.

    This is the benchmark's whole claim, so if a run demonstrates it the card
    should say which pair rather than leave a reader to spot it in the table.
    """
    for better in rows:
        for worse in rows:
            if better is worse:
                continue
            more_accurate = worse["pass_at_1"] > better["pass_at_1"]
            costs_more = worse["severity_weighted_loss"] > better["severity_weighted_loss"]
            if more_accurate and costs_more:
                floor = max(better["severity_weighted_loss"], 1e-9)
                ratio = worse["severity_weighted_loss"] / floor
                return (
                    f"`{worse['model_id']}` answers more of the suite correctly than "
                    f"`{better['model_id']}` ({worse['pass_at_1']:.3f} vs "
                    f"{better['pass_at_1']:.3f}) and still costs {ratio:.1f}x as much "
                    f"({worse['severity_weighted_loss']:.1f} vs "
                    f"{better['severity_weighted_loss']:.1f}), because its mistakes land "
                    "on higher-severity tasks. Accuracy and expected loss rank these two "
                    "in opposite orders, which is the reason this benchmark exists."
                )
    return ""


def _honest_limits(has_results: bool) -> str:
    limits = [
        "Tasks are generated, not observed. They are representative of "
        "back-office workloads by construction and citation, not by sampling.",
        "Severity costs are contested inputs, not constants. Every conclusion "
        "must be shown across a range of cost models.",
        "Ground truth is mechanical: the verifier recomputes the answer from "
        "`initial_state` alone and a task is only kept when it agrees, so "
        "verifier agreement is enforced at generation rather than measured.",
        "The decision rules are stated in each prompt. This benchmark measures "
        "whether a model applies a published policy to structured data, not "
        "whether it can guess an unpublished convention.",
    ]
    if has_results:
        limits.append(
            "Results are a single run on one seed against one gateway. Small n; "
            "treat the absolute numbers as indicative, not as a leaderboard."
        )
    return "\n".join(f"- {line}" for line in limits)


def _coverage_note(data: dict | None, eval_task_count: int) -> str:
    """Disclose how much of the shipped eval split the leaderboard actually covers.

    The published leaderboard is scored on a prefix subset of the eval split,
    not the whole thing, so the card has to say so out loud.
    """
    if not data or not data.get("models"):
        return (
            f"No leaderboard is published for this revision. All "
            f"{eval_task_count} eval tasks ship in `data/eval.jsonl`; none have "
            "published scores."
        )
    n = data.get("n_tasks")
    if not n:
        return (
            f"The published leaderboard does not record how many of the "
            f"{eval_task_count} shipped eval tasks it was scored on. Treat its "
            "coverage as unknown."
        )
    per_domain = n // len(DOMAINS)
    pct = 100 * n / eval_task_count if eval_task_count else 0
    lines = [
        f"The published leaderboard covers only the first {n} of the "
        f"{eval_task_count} eval tasks that ship in `data/eval.jsonl` "
        f"(~{pct:.0f}%). It is the prefix subset "
        f"`n_tasks={n}` ({per_domain}/domain), seed "
        f"{data.get('seed', EVAL_SEED)}, trials {data.get('trials', 1)}, "
        f"`max_steps={data.get('max_steps', 1)}`, `partial: "
        f"{str(bool(data.get('partial'))).lower()}`. The remaining eval tasks "
        "have no published scores.",
    ]
    if data.get("partial_note"):
        lines.append(f"Run note, verbatim: “{data['partial_note']}”")
    return "\n\n".join(lines)


def build_payload(out: Path, per_domain: int, artifacts: Path) -> dict:
    """Materialize everything the Hub repo will contain, under `out`."""
    exporter = _load_exporter()
    (out / "data").mkdir(parents=True, exist_ok=True)

    eval_tasks = _suite(EVAL_SEED, per_domain)
    train_tasks = _suite(TRAIN_SEED, per_domain)
    exporter.tasks_to_jsonl(eval_tasks, str(out / "data" / "eval.jsonl"))
    exporter.tasks_to_jsonl(train_tasks, str(out / "data" / "train.jsonl"))

    check = monitor_report(train_tasks, eval_tasks)
    certificate = {
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "signature_overlap": check["overlap"],
        "clean": check["false_fire"],
        "valid": check["overlap"] == 0.0,
    }
    if not certificate["valid"]:
        raise SystemExit(
            f"refusing to publish: train/eval signature overlap is "
            f"{check['overlap']}, expected 0.0"
        )
    (out / "contamination_certificate.json").write_text(
        json.dumps(certificate, indent=2) + "\n", encoding="utf-8"
    )

    leaderboard = artifacts / "leaderboard.json"
    results_table, generated_at = _results_table(leaderboard)
    board_data = (
        json.loads(leaderboard.read_text(encoding="utf-8"))
        if leaderboard.exists()
        else None
    )
    coverage_note = _coverage_note(board_data, len(eval_tasks))
    if generated_at:
        (out / "results").mkdir(exist_ok=True)
        shutil.copyfile(leaderboard, out / "results" / "leaderboard.json")
        # report.md is only shipped when it belongs to the same run. A
        # stopped run leaves the previous report in place, and the last one
        # written here was a stub whose losses are all 0.0000.
        report = artifacts / "report.md"
        if report.exists() and report.stat().st_mtime >= leaderboard.stat().st_mtime:
            shutil.copyfile(report, out / "results" / "report.md")

    weights = {
        cm: {band: load_cost_profile(cm).cost(Severity(band)) for band in SEVERITY_BANDS}
        for cm in COST_MODELS
    }
    (out / "cost_models.json").write_text(
        json.dumps(weights, indent=2, default=str) + "\n", encoding="utf-8"
    )

    card = exporter.build_dataset_card(
        benchmark_id="LossBench finance-v1",
        description=(
            "Severity-weighted expected-loss evaluation for agents that touch "
            "money. Three finance back-office domains, mechanical ground truth, "
            "and a contamination certificate. Models are ranked by what their "
            "mistakes cost, not by raw accuracy."
        ),
        license_name=LICENSE,
        task_count=len(eval_tasks) + len(train_tasks),
        domains=list(DOMAINS),
        severity_taxonomy=SEVERITY_BANDS,
        cost_model_ids=COST_MODELS,
        contamination_policy=(
            f"Train is seed {TRAIN_SEED}, eval is seed {EVAL_SEED}. A task "
            "signature is a SHA-256 over every field except `id`, `seed` and "
            "`signature`, so a renumbered copy of the same content still "
            f"collides. Measured overlap between the two splits is "
            f"{check['overlap']}, and publishing is refused if it is anything "
            "else. That signature detects training-set overlap against this "
            "dataset; it does not detect evaluation gaming.\n\n"
            "The labels (`gold`, `severity`) ship in plaintext in "
            "`data/eval.jsonl`. This is an open reference benchmark, not a "
            "hidden-answer-key benchmark: the eval set can be fingerprinted "
            "from the published data, and third-party leaderboard submissions "
            "should be treated as self-reported."
        ),
        reproducibility_notes=(
            "Every row is a pure function of (generator version, domain, seed, "
            "index). Regenerate with `generate_suite(domain, seed, n_tasks)` "
            "from https://github.com/caiotheodoro/lossbench. The same seed "
            "yields byte-identical tasks, which the repo enforces as a CI gate. "
            "Because identity is code-version dependent, cite a revision rather "
            "than `main`."
        ),
        contact="https://github.com/caiotheodoro/lossbench/issues",
        results_table=results_table,
        coverage_note=coverage_note,
        honest_limits=_honest_limits(bool(generated_at)),
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    # eval.yaml is intentionally NOT emitted. The only manifest shape this repo
    # can currently produce (id/dataset/task_types/metric/license/paper) is the
    # exact shape HF rejects at push-time validation -- see section 4.2 of
    # hf-publication-specs.md. The schema HF actually wants
    # (name/description/evaluation_framework/tasks[].id) needs `lossbench` added
    # to HF's `evaluation_framework` enum first, which is blocker B-3. Until B-3
    # clears we publish as a plain dataset with no eval.yaml rather than ship a
    # file that fails validation. Do not "fix" this by emitting the new schema
    # early -- the benchmark listing is still beta + allow-listed.
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--per-domain", type=int, default=400)
    parser.add_argument("--artifacts", type=Path, default=_REPO_ROOT / "artifacts")
    parser.add_argument("--out", type=Path, default=Path("/tmp/lossbench-hf"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the payload locally and stop before touching the Hub",
    )
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    certificate = build_payload(args.out, args.per_domain, args.artifacts)

    print(f"payload built in {args.out}")
    for path in sorted(args.out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(args.out)}  {path.stat().st_size:,} bytes")
    print(f"  contamination: overlap={certificate['signature_overlap']} valid=True")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(args.out),
        commit_message=(
            f"finance-v1: {certificate['eval_tasks']} eval / "
            f"{certificate['train_tasks']} train, overlap 0.0"
        ),
    )
    print(f"published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
