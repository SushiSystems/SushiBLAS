# diag.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Builds the sushicore Diagnostics object behind `sb config` and `sb env`."""

from __future__ import annotations

from pathlib import Path

from sushicore.diag import Diagnostics

from .. import config as config_module
from .. import console
from ..config import PROFILE
from ..env import create_build_env


def _build_env(cfg, build_dir: Path) -> dict[str, str]:
    """Returns the environment a build in *build_dir* runs under."""
    return create_build_env(console).load(cfg, build_dir)


K_DIAGNOSTICS = Diagnostics(
    profile=PROFILE,
    console=console,
    config=config_module,
    build_env=_build_env,
)
