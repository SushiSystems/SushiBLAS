# executable_lookup_memory.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Finds executables by name in a list held in memory, with no walk over a tree.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from sushicore.example_catalog import ExampleCatalog


@typing.final
@dataclass(frozen=True, slots=True)
class MemoryExecutableLookup:
    """Keeps the promise of :class:`ExecutableLookup` over names that sit directly under a tree.

    Args:
        names: The executables every tree holds, in the order a pick lists them.
    """

    names: tuple[str, ...] = ()

    def match(self, build_root: Path, query: str) -> Path | None:
        """Returns ``build_root / query`` when *query* is one of the names, else None."""
        return build_root / query if query in self.names else None

    def select(self, build_root: Path, console, *,
               examples: ExampleCatalog | None = None) -> Path | None:
        """Returns the first row of the pick under *build_root*; None when there is no row.

        With *examples* the rows are the names those examples register, or every name when none is.
        """
        names = self.names
        if examples is not None:
            registered = {example.name for example in examples.entries()}
            names = tuple(name for name in names if name in registered) or names
        return build_root / names[0] if names else None
