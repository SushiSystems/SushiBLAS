# discovery.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Builds the index of executables under SushiBLAS's build tree, skipping sibling checkouts."""

from __future__ import annotations

from sushicore.discovery import ExecutableIndex

from ..config import PROFILE


def create_executable_index() -> ExecutableIndex:
    """Returns an index that skips the sibling checkouts PROFILE names."""
    return ExecutableIndex(skip_dirs=PROFILE.sibling_skip_dirs())
