"""Entrypoint for the lossbench click CLI."""

from __future__ import annotations

import click

from lossbench.cli.commands import costs, decide, metrics, simulate, version


def cli() -> None:
    """Run the lossbench CLI (group name 'lossbench')."""
    main()


def main() -> None:
    """Build and invoke the 'lossbench' click group."""
    build_group()()


def build_group() -> click.Group:
    """Construct the 'lossbench' click group with all commands registered."""
    group = click.Group(name="lossbench")
    for command in (metrics, costs, decide, simulate, version):
        group.add_command(command)
    return group
