# test_help_screen.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""sb's help screen groups its commands under three headings and carries examples."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from sushiblas.cli import app

pytestmark = pytest.mark.small

K_GROUPS = {
    "Project": ["build", "test", "bench", "run", "clean", "package"],
    "Diagnostics": ["config", "env", "setup", "doctor", "link", "unlink"],
    "Documentation": ["docs", "backlog", "design"],
}


def _run(*args: str) -> str:
    """Return the output of one sb invocation, asserting it exited zero."""
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def _commands_by_heading(page: str) -> dict[str, list[str]]:
    """Return the first word of each row, keyed by the heading above it.

    A row starts two columns in; a wrapped description sits deeper and is skipped.
    """
    grouped: dict[str, list[str]] = {}
    heading = ""
    for line in page.splitlines():
        if line and not line.startswith(" "):
            heading = line.rstrip()
            grouped[heading] = []
        elif line.startswith("  ") and not line.startswith("   "):
            grouped[heading].append(line.split()[0])
    return grouped


def _last_heading(page: str) -> str:
    """Return the last line of *page* that starts in the first column: the heading of its last section."""
    return [line.rstrip() for line in page.splitlines() if line and not line.startswith(" ")][-1]


def test_root_help_lists_the_three_groups_in_order():
    lines = [line.rstrip() for line in _run("--help").splitlines()]
    positions = [lines.index(heading) for heading in K_GROUPS]
    assert positions == sorted(positions)


@pytest.mark.parametrize("heading", list(K_GROUPS))
def test_no_command_is_listed_under_a_group_it_does_not_belong_to(heading):
    assert set(_commands_by_heading(_run("--help"))[heading]) == set(K_GROUPS[heading])


@pytest.mark.parametrize(
    "name",
    [name for names in K_GROUPS.values() for name in names if name not in ("backlog", "design")],
)
def test_every_command_help_ends_with_examples(name):
    assert _last_heading(_run(name, "--help")) == "Examples"

