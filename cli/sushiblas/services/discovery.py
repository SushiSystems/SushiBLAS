"""Locate built executables under SushiBLAS's build tree.

The walk itself is shared -- see :class:`sushicore.discovery.ExecutableIndex`.
``sushiruntime`` is skipped because sushiblas builds it in-tree, and
its executables are not sushiblas's to offer.
"""

from __future__ import annotations

from pathlib import Path

from sushicore.discovery import ExecutableIndex

from .. import console

_INDEX = ExecutableIndex(skip_dirs=("sushiruntime",))

find_executables = _INDEX.find
match_by_name = _INDEX.match


def select_interactive(build_root: Path) -> Path | None:
    """Show a table of executables and prompt the user to pick one."""
    return _INDEX.select(build_root, console)
