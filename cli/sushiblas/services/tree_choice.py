# tree_choice.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the build tree a command works in: one per build type and sanitizer set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sushicore.build_tree import tree_dir

from .build_type import K_CMAKE_BUILD_TYPE, K_DEFAULT_BUILD_TYPE, BuildType
from .sanitizer import K_SANITIZE_TREE, Sanitizer


@dataclass(frozen=True, slots=True)
class TreeChoice:
    """Identifies one build tree by what it is built as.

    Args:
        build_type: The build type of the tree.
        sanitizer: The sanitizer set of the tree; None is the plain tree, same as NONE.
    """

    build_type: BuildType = K_DEFAULT_BUILD_TYPE
    sanitizer: Sanitizer | None = None

    @property
    def cmake_type(self) -> str:
        """Returns the CMAKE_BUILD_TYPE value of the tree."""
        return K_CMAKE_BUILD_TYPE[self.build_type]

    def directory(self, root: Path, requested: str | None = None) -> Path:
        """Returns the tree below *root*: build/<name>-<type>, or *requested* below *root*."""
        if requested:
            return root / requested
        return tree_dir(root, K_SANITIZE_TREE[self.sanitizer or Sanitizer.NONE], self.cmake_type)

    def build_command(self) -> str:
        """Returns the `sb build` command line that makes this tree."""
        command = f"sb build --type {self.build_type.value}"
        if self.sanitizer not in (None, Sanitizer.NONE):
            command += f" --sanitize {self.sanitizer.value}"
        return command
