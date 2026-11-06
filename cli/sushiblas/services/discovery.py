# discovery.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Builds the index of executables under SushiBLAS's build tree, skipping sibling checkouts."""

from __future__ import annotations

from sushicore.discovery import ExecutableIndex

from ..config import PROFILE


def create_executable_index() -> ExecutableIndex:
    """Returns an index that skips the sibling checkouts PROFILE names."""
    return ExecutableIndex(skip_dirs=PROFILE.sibling_skip_dirs())
