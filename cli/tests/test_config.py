# test_config.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for config-dir selection."""

from __future__ import annotations

import pytest

from sushiblas import config

pytestmark = pytest.mark.medium


@pytest.mark.parametrize("has_config", [True, False], ids=["config.toml present", "config.toml absent"])
def test_config_dir_is_the_cli_folder_with_or_without_a_config_file(tmp_path, has_config):
    """config_dir() is <root>/cli whether or not that folder holds config.toml."""
    (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    if has_config:
        (cli_dir / "config.toml").write_text("")
    assert config.config_dir(tmp_path) == cli_dir
