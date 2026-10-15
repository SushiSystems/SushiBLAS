# conftest.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Makes the real sushicore importable, and holds the size rules every CLI test follows.

See cli/tests/README.md.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _ensure_sushicore() -> None:
    """Make :mod:`sushicore` importable, or fail with a message that says how.

    An installed package wins. Failing that, it is taken from a SushiCore checkout
    beside this repository, at ``<siblings>/sushicore``.
    """
    if importlib.util.find_spec("sushicore") is not None:
        return

    repo = Path(__file__).resolve().parents[2]
    candidate = repo.parent / "sushicore"
    if (candidate / "sushicore" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        importlib.invalidate_caches()
        if importlib.util.find_spec("sushicore") is not None:
            return

    raise RuntimeError(
        "sushicore is not importable, so the sushiblas CLI cannot be imported. "
        "Install it with `pip install sushicore`, or place a SushiCore checkout "
        "next to this one."
    )


_ensure_sushicore()


K_SIZE_TIMEOUTS = {"small": 60, "medium": 300, "large": 900}


def _sizes_of(item: pytest.Item) -> list[str]:
    """Returns the size labels *item* carries, one entry per label."""
    return [mark.name for mark in item.iter_markers() if mark.name in K_SIZE_TIMEOUTS]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fails collection when a test carries no size or more than one, and gives each its timeout."""
    wrong = [f"{item.nodeid}: {_sizes_of(item) or 'no size'}"
             for item in items if len(_sizes_of(item)) != 1]
    if wrong:
        raise pytest.UsageError(f"every test carries exactly one size: {', '.join(wrong)}")
    for item in items:
        item.add_marker(pytest.mark.timeout(K_SIZE_TIMEOUTS[_sizes_of(item)[0]]))
