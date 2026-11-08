# test_provision_commands.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that sb carries the shared provision commands."""

from __future__ import annotations

from dataclasses import fields

import pytest
from typer.testing import CliRunner

from sushicore.provision.config import ProvisionSettings
from sushiblas.cli import app
from sushiblas.config import Config

K_PROVISION_COMMANDS = ("setup", "doctor", "link", "unlink")
K_SETUP_OPTIONS = ("--dry-run", "--toolchain", "--no-gpu")


def _absent_from(text: str, words: tuple[str, ...]) -> list[str]:
    """Returns the entries of *words* that *text* does not hold."""
    return [word for word in words if word not in text]


def _field_names(cls: type) -> set[str]:
    """Returns the names of the dataclass fields of *cls*."""
    return {field.name for field in fields(cls)}


@pytest.mark.small
class TestTheProvisionCommandsAreOffered:
    """What the help screens show of the shared provision commands, with no checkout."""

    def test_the_four_provision_commands_are_registered(self):
        """Assert setup, doctor, link and unlink appear in the help."""
        out = CliRunner().invoke(app, ["--help"]).output
        assert _absent_from(out, K_PROVISION_COMMANDS) == []

    def test_setup_takes_the_shared_options(self):
        """Assert setup offers --dry-run, --toolchain and --no-gpu."""
        out = CliRunner().invoke(app, ["setup", "--help"]).output
        assert _absent_from(out, K_SETUP_OPTIONS) == []

    def test_config_carries_every_field_provision_reads(self):
        """Assert every field of the provision settings is a field of the config."""
        assert _field_names(ProvisionSettings) - _field_names(Config) == set()


@pytest.mark.medium
class TestSetupReadsTheCheckout:
    """What setup decides from the sibling checkouts on disk."""

    def test_setup_stops_when_the_runtime_checkout_is_missing(self, tmp_path, monkeypatch):
        """Assert setup exits 2 and names sushiruntime when it has no checkout."""
        root = tmp_path / "sushiblas"
        (root / "cli").mkdir(parents=True)
        (root / "CMakeLists.txt").write_text("")
        (root / "cli" / "config.toml").write_text("")
        (root / "cli" / "sushistack.deps.toml").write_text(
            '[module]\ndepends_on = ["sushiruntime"]\n')
        monkeypatch.chdir(root)
        monkeypatch.setenv("SUSHISYSTEMS_HOME", str(tmp_path / "home"))
        monkeypatch.delenv("SUSHISTACK_HOME", raising=False)
        monkeypatch.delenv("SUSHIRUNTIME_DIR", raising=False)
        monkeypatch.delenv("SUSHISTACK_DEPS_DIR", raising=False)
        result = CliRunner().invoke(app, ["setup", "--dry-run"])
        assert result.exit_code == 2
        assert "sushiruntime" in result.output
