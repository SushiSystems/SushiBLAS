# package_source.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Names the port the build asks for the built packages of the Sushi libraries it links.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from ..config import Config


class PackageSource(Protocol):
    """Chooses, and builds when missing, the packages of the closure a build links."""

    def resolve(self, cfg: Config, root: Path, package_type: str,
                options: Mapping[str, tuple[str, ...]], report: Callable[[str], None],
                warn: Callable[[str], None], compiler: str):
        """Returns the packages of the closure: their names, definitions and run-time folders.

        Args:
            cfg: The resolved configuration.
            root: The project root.
            package_type: The package build type, such as ``release``.
            options: The extra ``build`` arguments of a library, by library name.
            report: Receives one ``reused`` or ``built`` line per library.
            warn: Receives the warnings of the resolver.
            compiler: The compiler of this build; the packages are built with it.

        Raises:
            sushicore.provision.sushi_packages.PackageError: A package cannot be chosen.
        """
        ...
