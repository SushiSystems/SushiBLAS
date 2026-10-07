# test_cli.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for sb's entry point, its root options and the commands it takes from sushicore."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from sushicore.root_options import version_line
from typer.testing import CliRunner

import sushiblas.console
from sushiblas import cli

K_VISIBLE_COMMANDS = [
    "build", "clean", "config", "docs", "docs bundle", "doctor", "env", "link", "run", "setup",
    "test", "unlink",
]
K_ENVIRONMENT = ("SUSHI_CLI_THEME", "SUSHI_CLI_ICONS", "SUSHI_CLI_COLOR", "SUSHISTACK_HOME")
K_DOCS_EXIT_CODE = 6


class _DocsService:
    """Stands in for the project service; docs returns a fixed code."""

    def docs(self) -> int:
        """Returns K_DOCS_EXIT_CODE."""
        return K_DOCS_EXIT_CODE


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """Returns a function that makes an empty checkout with the given cli/config.toml."""

    def make(config_text: str) -> Path:
        """Writes the checkout, enters it and forgets any console built before."""
        root = tmp_path / "sushiblas"
        (root / "cli").mkdir(parents=True)
        (root / "CMakeLists.txt").write_text("")
        (root / "cli" / "config.toml").write_text(config_text)
        monkeypatch.chdir(root)
        monkeypatch.setenv("SUSHISYSTEMS_HOME", str(tmp_path / "home"))
        for name in K_ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
        importlib.reload(sushiblas.console)
        return root

    yield make
    monkeypatch.undo()
    importlib.reload(sushiblas.console)


def _run_main(monkeypatch, capsys, *args: str) -> tuple[int, list[str]]:
    """Returns the exit code of main() and the non-empty lines it wrote."""
    monkeypatch.setattr(sys, "argv", ["sb", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    captured = capsys.readouterr()
    lines = [line for line in (captured.out + captured.err).splitlines() if line.strip()]
    return exit_info.value.code, lines


@pytest.mark.parametrize("command", ["config", "clean"])
def test_main_reports_a_malformed_config_file_as_one_line(checkout, monkeypatch, capsys, command):
    """A config.toml that is not TOML ends in exit 1 and one line naming the file."""
    checkout("[tool\nbroken")
    code, lines = _run_main(monkeypatch, capsys, command)
    assert code == 1
    assert len(lines) == 1, lines
    assert "config.toml" in lines[0]
    assert "Traceback" not in lines[0]


@pytest.mark.parametrize("command", ["config", "clean"])
def test_main_reports_an_unknown_theme_as_one_line(checkout, monkeypatch, capsys, command):
    """A theme sushicore does not know ends in exit 1 and one line naming it."""
    checkout('[cli]\ntheme = "no-such-theme"\n')
    code, lines = _run_main(monkeypatch, capsys, command)
    assert code == 1
    assert len(lines) == 1, lines
    assert "no-such-theme" in lines[0]


def test_main_exits_zero_after_a_command_that_succeeds(checkout, monkeypatch, capsys):
    """A command that works leaves exit code 0 through main()."""
    checkout("")
    code, _ = _run_main(monkeypatch, capsys, "clean")
    assert code == 0


def test_version_prints_the_distribution_and_its_version():
    """--version prints one line naming sushiblas-cli and exits 0."""
    result = CliRunner().invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == version_line("sushiblas-cli")


def test_describe_lists_every_visible_command_and_no_hidden_one():
    """--describe prints the catalogue of the commands the help screen shows."""
    result = CliRunner().invoke(cli.app, ["--describe"])
    assert result.exit_code == 0
    document = json.loads(result.output)
    assert document["program"] == "sb"
    assert [command["name"] for command in document["commands"]] == K_VISIBLE_COMMANDS


def test_env_takes_the_short_form_of_all():
    """env comes from sushicore, which spells --all as -a too."""
    result = CliRunner().invoke(cli.app, ["env", "--help"])
    assert result.exit_code == 0
    assert "-a, --all" in result.output


def test_config_and_env_are_not_declared_in_this_package():
    """The two diagnostic commands are registered from sushicore, not written here."""
    registered = {info.name: info.callback.__module__ for info in cli.app.registered_commands}
    assert registered["config"] == "sushicore.diag_commands"
    assert registered["env"] == "sushicore.diag_commands"


def test_docs_runs_the_api_build_and_bundle_is_its_subcommand(monkeypatch, tmp_path):
    """`sb docs` still builds the reference, and `sb docs bundle` asks for a release."""
    monkeypatch.setattr(cli, "_service", _DocsService)
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    runner = CliRunner()
    built = runner.invoke(cli.app, ["docs"])
    assert built.exit_code == K_DOCS_EXIT_CODE
    assert "Doxygen Generation" in built.output
    assert runner.invoke(cli.app, ["doxygen"]).exit_code == K_DOCS_EXIT_CODE
    missing = runner.invoke(cli.app, ["docs", "bundle"])
    assert missing.exit_code == 2
    assert "--release" in missing.output


def test_the_api_source_reads_the_xml_doxygen_writes_under_the_site_folder(monkeypatch, tmp_path):
    """The bundle takes the API reference from docs/api-site/xml below the repository root."""
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    assert cli._api_source().xml_dir == tmp_path / "docs" / "api-site" / "xml"
