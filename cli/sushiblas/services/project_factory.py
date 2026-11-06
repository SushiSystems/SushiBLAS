# project_factory.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Wires a ProjectService to sushicore's runner, CMake driver and this checkout."""

from __future__ import annotations

from sushicore.cmake_driver import CMakeDriver
from sushicore.proc import Runner

from ..config import PROFILE, find_project_root, load_config
from ..env import create_build_env
from .discovery import create_executable_index
from .project import ProjectService


def create_project_service(console) -> ProjectService:
    """Returns a ProjectService that reports progress through *console*."""
    runner = Runner(console, PROFILE.program, catch_interrupt=True)
    return ProjectService(
        console=console,
        runner=runner,
        driver=CMakeDriver(console, runner),
        index=create_executable_index(),
        build_env=create_build_env(console),
        find_root=find_project_root,
        load_config=load_config,
    )
