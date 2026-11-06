# test_config.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for project-root resolution and config-dir selection."""

from __future__ import annotations

import pytest

from sushiblas import config

pytestmark = pytest.mark.medium


def test_config_dir_prefers_cli(tmp_path):
    """config_dir() returns <root>/cli when it holds config.toml."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    (cli_dir / "config.toml").write_text("")
    assert config.config_dir(tmp_path) == cli_dir


def test_config_dir_defaults_to_cli_when_absent(tmp_path):
    """With no config.toml present, config_dir() still defaults to <root>/cli."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    assert config.config_dir(tmp_path) == tmp_path / "cli"
