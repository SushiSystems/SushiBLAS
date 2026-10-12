# test_help_screen.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""sb's help screen groups its commands under three headings and carries examples."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import sushiblas
from sushiblas.cli import app

K_GROUPS = {
    "Project": ["build", "test", "run", "clean"],
    "Diagnostics": ["config", "env", "setup", "doctor", "link", "unlink"],
    "Documentation": ["docs"],
}
K_RAW_COLOUR = re.compile(r"\[/?(green|red|yellow|cyan|blue|dim)\b")
K_PACKAGE_DIR = Path(sushiblas.__file__).parent


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


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Return every string literal of a Python file with its line, f-string parts included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_root_help_lists_the_three_groups_in_order():
    lines = [line.rstrip() for line in _run("--help").splitlines()]
    positions = [lines.index(heading) for heading in K_GROUPS]
    assert positions == sorted(positions)


@pytest.mark.parametrize("heading", list(K_GROUPS))
def test_every_command_sits_under_its_own_group(heading):
    listed = _commands_by_heading(_run("--help"))[heading]
    assert [name for name in listed if name in K_GROUPS[heading]] == K_GROUPS[heading]


def test_no_command_is_listed_under_a_group_it_does_not_belong_to():
    grouped = _commands_by_heading(_run("--help"))
    for heading, names in K_GROUPS.items():
        assert set(grouped[heading]) == set(names)


def test_a_command_help_carries_examples():
    out = _run("build", "--help")
    assert "Examples" in out and "sb build --type debug" in out


def test_a_command_with_arguments_carries_examples():
    out = _run("run", "--help")
    assert "Examples" in out and "sb run quickstart_demo" in out


@pytest.mark.parametrize("name", [name for names in K_GROUPS.values() for name in names])
def test_every_command_help_ends_with_examples(name):
    assert "Examples" in _run(name, "--help")


def test_no_source_string_names_a_raw_colour():
    offenders = {
        f"{path.relative_to(K_PACKAGE_DIR).as_posix()}:{line}"
        for path in K_PACKAGE_DIR.rglob("*.py")
        for line, text in _string_literals(path)
        if K_RAW_COLOUR.search(text)
    }
    assert sorted(offenders) == []
