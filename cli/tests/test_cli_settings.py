# test_cli_settings.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for what sb's root options and its appearance settings decide, with no checkout."""

from __future__ import annotations

import json
import re

import pytest
from sushicore.config import load_appearance
from typer.testing import CliRunner

from sushiblas import cli

pytestmark = pytest.mark.small

#: The groups sushicore adds whole; their commands are sushicore's to list, so only the group is checked.
K_SUSHICORE_GROUPS = ("backlog", "design")
K_VISIBLE_COMMANDS = [
    "bench", "build", "clean", "config", "docs", "docs bundle", "doctor", "env", "link", "package", "run",
    "setup", "test", "unlink",
]
K_VERSION_LINE = re.compile(r"sushiblas-cli \d+\.\d+\.\d+\S*")


def test_version_prints_the_distribution_and_its_version():
    """--version prints one line naming sushiblas-cli and a version, and exits 0."""
    result = CliRunner().invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert K_VERSION_LINE.fullmatch(result.output.strip())


def test_describe_lists_every_visible_command_and_no_hidden_one():
    """--describe prints the catalogue of the commands the help screen shows."""
    result = CliRunner().invoke(cli.app, ["--describe"])
    assert result.exit_code == 0
    document = json.loads(result.output)
    assert document["program"] == "sb"
    names = [command["name"] for command in document["commands"]]
    assert [name for name in names if name.split()[0] not in K_SUSHICORE_GROUPS] == K_VISIBLE_COMMANDS
    assert {name.split()[0] for name in names} >= set(K_SUSHICORE_GROUPS)


def test_env_takes_the_short_form_of_all():
    """env comes from sushicore, which spells --all as -a too."""
    result = CliRunner().invoke(cli.app, ["env", "--help"])
    assert result.exit_code == 0
    assert "-a, --all" in result.output


def test_an_unknown_theme_settles_on_the_default_with_one_warning_naming_it(monkeypatch):
    """A theme sushicore does not know keeps the command running at the default, with one warning."""
    monkeypatch.setenv("SUSHI_CLI_THEME", "no-such-theme")
    spec = load_appearance([])
    assert spec.theme == "default"
    assert len(spec.warnings) == 1
    assert "Unknown CLI theme 'no-such-theme'" in spec.warnings[0]
