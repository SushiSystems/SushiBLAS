# conftest.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Registers the size marks of `testing` and sets each test's timeout from its size."""

from __future__ import annotations

import pytest

K_TIMEOUTS_IN_SECONDS = {"small": 60, "medium": 300, "large": 900}


def pytest_configure(config: pytest.Config) -> None:
    """Registers one mark for each size."""
    for size, seconds in K_TIMEOUTS_IN_SECONDS.items():
        config.addinivalue_line("markers", f"{size}: a {size} test, timed out after {seconds} s")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Gives each test the timeout of its size and fails collection of a test that carries none."""
    for item in items:
        sizes = [size for size in K_TIMEOUTS_IN_SECONDS if item.get_closest_marker(size)]
        if len(sizes) != 1:
            raise pytest.UsageError(f"{item.nodeid} must carry exactly one size mark, found {sizes}")
        item.add_marker(pytest.mark.timeout(K_TIMEOUTS_IN_SECONDS[sizes[0]]))
