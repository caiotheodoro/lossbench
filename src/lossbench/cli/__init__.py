"""LossBench CLI (P1.9): click entrypoint, groups, and commands."""

from lossbench.cli.commands import costs, decide, metrics, simulate, version
from lossbench.cli.main import cli

__all__ = ["cli", "costs", "decide", "metrics", "simulate", "version"]
