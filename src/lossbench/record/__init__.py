"""Trajectory recorder: OTel span emission plus OpenAI-compatible proxy mode."""

from lossbench.record.proxy import run_proxy
from lossbench.record.recorder import TrajectoryRecorder, event_from_trace

__all__ = ["TrajectoryRecorder", "event_from_trace", "run_proxy"]
