# env.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Builds the snapshotter of the SushiBLAS build environment on sushicore's StackBuildEnv.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from sushicore.build_env import StackBuildEnv

from .config import PROFILE, find_project_root


def create_build_env(console) -> StackBuildEnv:
    """Returns the snapshotter of the environment build, test and run subprocesses inherit."""
    return StackBuildEnv(profile=PROFILE, console=console, find_root=find_project_root)
