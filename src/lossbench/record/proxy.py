"""Minimal OpenAI-compatible proxy mode that records decisions."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from openai import OpenAI

from lossbench.record.recorder import TrajectoryRecorder
from lossbench.schema import DecisionEvent, DecisionKind


def run_proxy(
    prompt: str,
    *,
    base_url: str,
    model_id: str,
    api_key: str,
    recorder: TrajectoryRecorder,
    trajectory_id: str,
    task_id: str,
    policy_id: str,
    cost_model_id: str,
) -> tuple[dict[str, Any], str]:
    """Make one OpenAI-compatible chat request and record the decision.

    A DecisionEvent with decision ALLOW is recorded when the response
    succeeds; ABSTAIN is recorded on timeout or HTTP errors. The prompt is
    hashed with SHA-256 into prompt_hash and input_snapshot_hash, and token
    counts are approximate estimates (len(text) // 4). Returns
    (event.model_dump(), response_text). Raises RuntimeError when the API
    call fails and no recorder is attached.
    """
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    started = time.perf_counter()
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        text = response.choices[0].message.content or ""
        decision = DecisionKind.ALLOW
    except Exception as exc:
        if recorder is None:
            raise RuntimeError("proxy API call failed and no recorder is attached") from exc
        text = ""
        decision = DecisionKind.ABSTAIN
    latency_ms = (time.perf_counter() - started) * 1000.0
    event = DecisionEvent(
        event_id=uuid.uuid4().hex,
        trace_id=uuid.uuid4().hex,
        trajectory_id=trajectory_id,
        task_id=task_id,
        timestamp=datetime.now(UTC),
        input_snapshot_hash=prompt_hash,
        prompt_hash=prompt_hash,
        model_id=model_id,
        token_usage={
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(text) // 4,
        },
        latency_ms=latency_ms,
        decision=decision,
        policy_id=policy_id,
        cost_model_id=cost_model_id,
    )
    if recorder is not None:
        recorder.record_decision(event)
    return event.model_dump(), text
