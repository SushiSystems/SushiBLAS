# test_cli.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for sb's entry point, its root options and the commands it takes from sushicore."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import pytest
from sushicore.console_memory import RecordingConsole
from typer.testing import CliRunner

import sushiblas.console
from sushiblas import cli
from sushiblas.command_context import default_context, using

pytestmark = pytest.mark.medium

K_ENVIRONMENT = ("SUSHI_CLI_THEME", "SUSHI_CLI_ICONS", "SUSHI_CLI_COLOR", "SUSHISTACK_HOME")
K_DOCS_EXIT_CODE = 6
K_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
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


def test_main_exits_zero_after_a_command_that_succeeds(checkout, monkeypatch, capsys):
    """A command that works leaves exit code 0 through main()."""
    checkout("")
    code, _ = _run_main(monkeypatch, capsys, "clean")
    assert code == 0


def test_docs_builds_the_api_reference_and_returns_the_exit_code_of_doxygen(commands):
    """`sb docs` runs Doxygen under its header and ends with Doxygen's exit code."""
    commands.runner.script(K_DOCS_EXIT_CODE)
    assert commands.invoke("docs").exit_code == K_DOCS_EXIT_CODE
    assert "Doxygen Generation" in commands.console.texts("header")


def test_docs_bundle_asks_for_a_release(commands):
    """`sb docs bundle` without --release is a usage error that names the option and runs nothing."""
    missing = commands.invoke("docs", "bundle")
    assert missing.exit_code == 2
    assert "--release" in K_ANSI_ESCAPE.sub("", missing.output)
    assert commands.runner.commands == []


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


def test_docs_bundle_writes_the_archive_of_the_checkout_it_runs_in(tmp_path, monkeypatch):
    """Writes the archive and its digest file at the checkout's root and reports both lines."""
    _published_checkout(tmp_path)
    monkeypatch.chdir(tmp_path / "docs" / "guides")
    console = RecordingConsole()
    context = replace(default_context(), console=console)
    with using(context):
        result = CliRunner().invoke(cli.app, ["docs", "bundle", "--release", "1.2.3"])
    said = console.texts("success")
    assert result.exit_code == 0, result.output
    bundle = tmp_path / "build" / "docs" / "bundle"
    assert (bundle / K_ARCHIVE).is_file()
    assert (bundle / (K_ARCHIVE + ".sha256")).is_file()
    assert len(said) == 2, said
    assert said[0].endswith(f"{K_ARCHIVE} (1 pages)")
    assert said[1].startswith("sha256 ")


def test_the_api_source_reads_the_xml_doxygen_writes_under_the_site_folder(commands):
    """Names build/docs/api-site/xml below the repository root."""
    with using(commands.context):
        assert cli._api_source().xml_dir == commands.root / "build" / "docs" / "api-site" / "xml"
