"""Agent-mode evaluation harness: stub-deterministic multi-step episodes."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from lossbench.cache.store import ResponseCache
from lossbench.runners.base import ModelRunner, RunnerResult
from lossbench.schema import DecisionEvent, DecisionKind, Task
from lossbench.scoring.passk import outcome_verified_pass_at_k, pass_k_reliability

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_VERIFIER_MODULES = {
    "verifier_reconciliation": "lossbench.generate.reconciliation",
    "verifier_payment_repair": "lossbench.generate.payment_repair",
    "verifier_settlement": "lossbench.generate.settlement",
}


@dataclass
class TrialResult:
    """Outcome of one agent episode on one task."""

    task_id: str
    model_id: str
    success: bool
    events: list[DecisionEvent]
    duration_ms: float
    cost: float
    parse_ok: bool = True
    errored: bool = False


def domain_verifier(task: Task, outcome: dict[str, Any]) -> bool:
    """Dispatch to the task's registered domain verifier by task.verifier name.

    Supports verifier_reconciliation, verifier_payment_repair, and
    verifier_settlement, imported lazily from the corresponding
    lossbench.generate module so domains whose generator has not landed
    still resolve to a clear ValueError instead of an ImportError.
    """
    module_name = _VERIFIER_MODULES.get(task.verifier)
    if module_name is None:
        raise ValueError(f"unknown verifier name '{task.verifier}'")
    try:
        module = importlib.import_module(module_name)
        verifier = getattr(module, task.verifier)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"verifier '{task.verifier}' is not available") from exc
    return bool(verifier(task, outcome))


def summarize_suite(results: Sequence[TrialResult]) -> dict[str, Any]:
    """Aggregate trial outcomes into the documented summary dictionary.

    Returns {"tasks", "trials", "pass_at_1", "pass_at_k", "pass_k",
    "total_cost", "mean_duration_ms", "false_success_rate",
    "false_success_applicable", "parse_rate", "error_rate"}: trials is the
    per-task trial count, pass_at_1/pass_at_k/pass_k are computed over k =
    trials per task.

    ``false_success_rate`` is always ``None`` and ``false_success_applicable``
    always ``False`` here: EvalHarness is a SELF_VERIFYING trajectory source
    (every event carries a gold-verified observed_outcome), so the
    false-success detector — "did the agent claim done with nothing
    verifying it?" — has no trajectory it can fire on and its rate is a
    structural constant, not a measurement. See
    ``lossbench.scoring.false_success`` for the CLAIM_THEN_VERIFY adapters
    where the metric is real.
    """
    per_task: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        per_task[result.task_id].append(result.success)
    tasks = len(per_task)
    if tasks == 0:
        return {
            "tasks": 0,
            "trials": 0,
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "pass_k": 0.0,
            "total_cost": 0.0,
            "mean_duration_ms": 0.0,
            "false_success_rate": None,
            "false_success_applicable": False,
            "parse_rate": 0.0,
            "error_rate": 0.0,
        }
    trials = len(results) // tasks
    matrix = [per_task[task_id] for task_id in sorted(per_task)]
    total_cost = sum(result.cost for result in results)
    mean_duration_ms = sum(result.duration_ms for result in results) / len(results)
    return {
        "tasks": tasks,
        "trials": trials,
        "pass_at_1": outcome_verified_pass_at_k(matrix, 1),
        "pass_at_k": outcome_verified_pass_at_k(matrix, trials),
        "pass_k": pass_k_reliability(matrix, trials),
        "total_cost": total_cost,
        "mean_duration_ms": mean_duration_ms,
        "false_success_rate": None,
        "false_success_applicable": False,
        "parse_rate": sum(r.parse_ok for r in results) / len(results),
        "error_rate": sum(r.errored for r in results) / len(results),
    }


class EvalHarness:
    """Run agent episodes against tasks and aggregate them with pass^k."""

    def __init__(
        self,
        runner: ModelRunner,
        cache: ResponseCache | None = None,
        max_steps: int = 8,
    ) -> None:
        self.runner = runner
        self.cache = cache
        self.max_steps = max_steps

    def run_task(self, task: Task, seed: int = 0) -> TrialResult:
        """Run one agent episode: prompt, corrective follow-ups, final decision.

        Deterministic for a given seed when the runner is a stub or the
        response cache supplies every step's answer. Step 0 uses
        task.prompt; a step whose outcome fails domain_verifier triggers a
        corrective follow-up, up to max_steps total steps. The last
        executed step carries the final ALLOW decision; success requires
        its outcome to match task.gold. Each step is cached via
        cache_key(model_id, prompt_hash, params, seed, input_hash) and a
        hit reuses the stored response without calling the runner.
        """
        start = time.perf_counter()
        input_hash = _sha256(_canonical_json(task.initial_state))
        events: list[DecisionEvent] = []
        prompt = task.prompt
        success = False
        parse_ok = False
        errored = False
        for step in range(self.max_steps):
            prompt_hash = _sha256(prompt)
            params = {"task_id": task.id, "step": step}
            try:
                result = self._decide(task, prompt, prompt_hash, params, seed, input_hash)
            except Exception:  # noqa: BLE001 - provider failures end this trial only
                errored = True
                break
            parsed = parse_outcome(result.text)
            parse_ok = parsed is not None
            outcome = parsed if parsed is not None else {}
            matched = parse_ok and domain_verifier(task, outcome)
            success = matched
            events.append(
                self._build_event(
                    task,
                    result,
                    prompt_hash,
                    input_hash,
                    seed,
                    step,
                    outcome,
                    matched,
                    matched or step == self.max_steps - 1,
                )
            )
            if matched:
                break
            prompt = _corrective_prompt(task)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return TrialResult(
            task_id=task.id,
            model_id=self.runner.name,
            success=success,
            events=events,
            duration_ms=duration_ms,
            cost=sum(event.model_cost for event in events),
            parse_ok=parse_ok,
            errored=errored,
        )

    def run_suite(
        self, tasks: Sequence[Task], trials: int = 3, seed: int = 0
    ) -> list[TrialResult]:
        """Run `trials` independent episodes per task with pass^k semantics.

        Trial t of each task runs with seed + t, so every episode is
        distinct yet the whole suite is deterministic for a given seed.
        """
        results: list[TrialResult] = []
        for task in tasks:
            for trial in range(trials):
                results.append(self.run_task(task, seed=seed + trial))
        return results

    def _decide(
        self,
        task: Task,
        prompt: str,
        prompt_hash: str,
        params: dict[str, Any],
        seed: int,
        input_hash: str,
    ) -> RunnerResult:
        if self.cache is None:
            return self.runner.decide(prompt, **params, seed=seed)
        key = self.cache.cache_key(
            self.runner.name, prompt_hash, params, seed, input_hash
        )
        cached = self.cache.get(key)
        if cached is not None:
            return RunnerResult(
                text=cached,
                model_id=self.runner.name,
                latency_ms=0.0,
                # Deterministic synthesis: matches the stub's own accounting,
                # so warm and cold runs emit byte-identical events. Real
                # token usage is not cached (documented limitation).
                token_usage={
                    "prompt_tokens": max(1, len(prompt.split())),
                    "completion_tokens": max(1, len(cached.split())),
                },
                cost=0.0,
                raw={},
            )
        result = self.runner.decide(prompt, **params, seed=seed)
        self.cache.put(key, result.text)
        return result

    def _build_event(
        self,
        task: Task,
        result: RunnerResult,
        prompt_hash: str,
        input_hash: str,
        seed: int,
        step: int,
        outcome: dict[str, Any],
        matched: bool,
        is_final: bool,
    ) -> DecisionEvent:
        timestamp = _EPOCH + timedelta(seconds=(abs(seed) % (1 << 24)) * 60 + step)
        confidence = _confidence(outcome, matched)
        enriched = dict(outcome)
        enriched.setdefault("severity", task.severity.value)
        if is_final:
            enriched["error"] = not matched
        return DecisionEvent(
            event_id=f"evt-{result.model_id}-{task.id}-{seed}-{step}",
            trace_id=f"trace-{result.model_id}-{task.id}-{seed}",
            trajectory_id=f"traj-{result.model_id}-{task.id}-{seed}",
            task_id=task.id,
            timestamp=timestamp,
            created_at=timestamp,
            input_snapshot_hash=input_hash,
            prompt_hash=prompt_hash,
            model_id=result.model_id,
            decision=DecisionKind.ALLOW if is_final else DecisionKind.VERIFY,
            observed_outcome=enriched,
            risk_features={"calibrated_p": confidence},
            calibrated_probability=confidence,
            policy_id=task.policy_id,
            cost_model_id=task.cost_model_ref,
            token_usage=dict(result.token_usage),
            latency_ms=0.0,
            model_cost=result.cost,
        )


def _confidence(outcome: dict[str, Any], matched: bool) -> float:
    """The model's own calibrated probability, when it gave a usable one.

    Falls back to a correctness-derived stand-in so older runners and
    unparseable answers still produce an event, but that fallback is not a
    calibration measurement and ECE computed over it is meaningless.
    """
    raw = outcome.get("confidence")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.9 if matched else 0.1
    value = float(raw)
    if value != value or not 0.0 <= value <= 1.0:  # NaN or out of range
        return 0.9 if matched else 0.1
    return value


def parse_outcome(text: str) -> dict[str, Any] | None:
    """Extract the answer object from a model response, or None on failure.

    Accepts a bare JSON object, a fenced ```json block, and a reasoning
    preamble ending in </think>. Returns None when no JSON object can be
    recovered; an unparseable answer is a parse miss scored as a failure,
    never silently coerced into a verdict.
    """
    body = text.split("</think>")[-1].strip()
    fence = _FENCE_RE.search(body)
    if fence is not None:
        body = fence.group(1).strip()
    for candidate in (body, _first_object(body)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_object(text: str) -> str:
    """Longest balanced {...} span in text, or "" when there is none."""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _corrective_prompt(task: Task) -> str:
    """Re-ask with the full task, not just a complaint.

    task.prompt already carries the input data and the output contract, so
    resending it verbatim keeps the retry answerable.
    """
    return (
        f"Your previous answer for task '{task.id}' did not pass the domain "
        f"verifier. Re-answer, reporting the correct outcome.\n\n{task.prompt}"
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
