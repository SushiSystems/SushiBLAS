# build_environment.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Names the port the project service reads the environment of a build, test or run through.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import Config


class BuildEnvironment(Protocol):
    """Gives the environment the subprocesses of one build tree inherit.

    ``sushicore.build_env.StackBuildEnv`` snapshots the toolchain shell;
    :class:`MemoryBuildEnvironment` hands back fixed variables.
    """

    def load(self, cfg: Config, build_dir: Path) -> dict[str, str]:
        """Returns a new mapping of the variables a subprocess of *build_dir* runs under."""
        ...
