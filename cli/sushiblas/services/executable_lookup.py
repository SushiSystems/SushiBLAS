# executable_lookup.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Names the port the project service finds a built executable through.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ExecutableLookup(Protocol):
    """Finds the executables of a build tree.

    ``sushicore.discovery.ExecutableIndex`` walks the tree on disk;
    :class:`MemoryExecutableLookup` answers from a list of names.
    """

    def match(self, build_root: Path, query: str) -> Path | None:
        """Returns the executable *query* names under *build_root*, or None."""
        ...

    def select(self, build_root: Path, console) -> Path | None:
        """Returns the executable the user picks under *build_root*, or None when there is none."""
        ...
