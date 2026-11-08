# build_environment_memory.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Hands every build tree the same environment, held in memory, with no shell run.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config


@typing.final
@dataclass(frozen=True, slots=True)
class MemoryBuildEnvironment:
    """Keeps the promise of :class:`BuildEnvironment` with fixed variables.

    Args:
        variables: The variables every tree's subprocesses run under.
    """

    variables: Mapping[str, str] = field(default_factory=dict)

    def load(self, cfg: Config, build_dir: Path) -> dict[str, str]:
        """Returns a new copy of the variables, so a caller's edit never reaches the next load."""
        return dict(self.variables)
