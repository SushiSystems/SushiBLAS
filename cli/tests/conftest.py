# conftest.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Makes the real sushicore importable before the SushiBLAS CLI tests are collected.

See cli/tests/README.md.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


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
