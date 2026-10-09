# test_cli.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for sb's entry point, its root options and the commands it takes from sushicore."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
K_GIT = (
    "git", "-c", "user.name=tests", "-c", "user.email=tests@example.invalid",
    "-c", "commit.gpgsign=false",
)
K_PUBLISH_LIST = (
    'name = "sample"\n'
    'title = "Sample"\n'
    'summary = "A sample repository."\n'
    'sections = ["guides"]\n'
    'api = false\n'
)
K_ARCHIVE = "docs-bundle-1.2.3.tar.gz"
K_DOXYFILE = Path(__file__).resolve().parents[2] / ".config" / "doxygen" / "Doxyfile"


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


def _published_checkout(root: Path) -> None:
    """Writes the CLI's root marker, a one-page docs/ tree and a git commit that holds them."""
    guides = root / "docs" / "guides"
    guides.mkdir(parents=True)
    (root / cli.PROFILE.root_marker).write_text("", encoding="utf-8")
    (root / "docs" / "publish.toml").write_text(K_PUBLISH_LIST, encoding="utf-8")
    (root / "docs" / "README.md").write_text(
        "# Sample\n\n- [Using](guides/USING.md)\n", encoding="utf-8")
    (guides / "USING.md").write_text("# Using\n\nRun it.\n", encoding="utf-8")
    for arguments in (("init", "--quiet"), ("add", "--all"),
                      ("commit", "--quiet", "--message", "docs")):
        subprocess.run([*K_GIT, *arguments], cwd=root, check=True, capture_output=True)


def _doxyfile_value(key: str, default: str) -> str:
    """Returns the value the repository's Doxyfile gives *key*, or *default* when it gives none."""
    for line in K_DOXYFILE.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return value.strip().strip('"')
    return default


def test_docs_bundle_writes_the_archive_of_the_checkout_it_runs_in(tmp_path, monkeypatch):
    """Writes the archive and its digest file at the checkout's root and reports both lines."""
    _published_checkout(tmp_path)
    monkeypatch.chdir(tmp_path / "docs" / "guides")
    said: list[str] = []
    monkeypatch.setattr(cli, "console", SimpleNamespace(success=said.append))
    result = CliRunner().invoke(cli.app, ["docs", "bundle", "--release", "1.2.3"])
    assert result.exit_code == 0, result.output
    bundle = tmp_path / "build" / "docs" / "bundle"
    assert (bundle / K_ARCHIVE).is_file()
    assert (bundle / (K_ARCHIVE + ".sha256")).is_file()
    assert len(said) == 2, said
    assert said[0].endswith(f"{K_ARCHIVE} (1 pages)")
    assert said[1].startswith("sha256 ")


def test_the_api_source_reads_the_xml_doxygen_writes_under_the_site_folder(monkeypatch, tmp_path):
    """Names `<OUTPUT_DIRECTORY>/<XML_OUTPUT>` of the Doxyfile, below the repository root."""
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    written = Path(_doxyfile_value("OUTPUT_DIRECTORY", "")) / _doxyfile_value("XML_OUTPUT", "xml")
    assert cli._api_source().xml_dir == tmp_path / written
