# package_source_memory.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Hands a build fixed packages held in memory and keeps what it asked for, with no package store.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config


@dataclass(frozen=True, slots=True)
class MemoryPackageSet:
    """Packages held as plain lists, with no store behind them.

    Args:
        library_names: The library names, in dependency order.
        cmake_definitions: The ``-D<Package>_DIR=`` flags of the closure.
        folders: The run-time folders, the last-resolved library first.
        left_out: The libraries of the closure the consumer asked not to resolve.
    """

    library_names: tuple[str, ...] = ()
    cmake_definitions: tuple[str, ...] = ()
    folders: tuple[str, ...] = ()
    left_out: tuple[str, ...] = ()

    def names(self) -> list[str]:
        """Returns the library names."""
        return list(self.library_names)

    def definitions(self) -> list[str]:
        """Returns the config folder definitions."""
        return list(self.cmake_definitions)

    def run_dirs(self, platform: str) -> list[str]:
        """Returns the run-time folders, whatever the platform."""
        return list(self.folders)


@dataclass(frozen=True, slots=True)
class PackageRequest:
    """One call a :class:`MemoryPackageSource` answered.

    Args:
        package_type: The package build type asked for.
        options: The extra ``build`` arguments asked for, by library name.
        compiler: The compiler asked for.
    """

    package_type: str
    options: Mapping[str, tuple[str, ...]]
    compiler: str


@typing.final
@dataclass(slots=True)
class MemoryPackageSource:
    """Keeps the promise of :class:`PackageSource` in memory: every request gets the same packages.

    Args:
        packages: What every request resolves to.
        requests: The requests answered so far, in order.
    """

    packages: MemoryPackageSet = field(default_factory=MemoryPackageSet)
    requests: list[PackageRequest] = field(default_factory=list)

    def resolve(self, cfg: Config, root: Path, package_type: str,
                options: Mapping[str, tuple[str, ...]], report: Callable[[str], None],
                warn: Callable[[str], None], compiler: str) -> MemoryPackageSet:
        """Records the request and returns the packages this source holds."""
        self.requests.append(PackageRequest(package_type, dict(options), compiler))
        return self.packages
