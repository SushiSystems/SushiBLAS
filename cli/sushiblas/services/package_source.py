# package_source.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Hands the build the built packages of the Sushi libraries it links: the port and its two adapters.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sushicore.provision.sushi_packages import PackageSet

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


class StorePackageSource:
    """Resolves the packages through the package store of the configuration."""

    def resolve(self, cfg: Config, root: Path, package_type: str,
                options: Mapping[str, tuple[str, ...]], report: Callable[[str], None],
                warn: Callable[[str], None], compiler: str) -> PackageSet:
        """Returns the built packages of ``cfg.sushi_packages``, building a missing one."""
        return cfg.sushi_packages(root, package_type, options, report=report, warn=warn,
                                  compiler=compiler)


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


@dataclass(slots=True)
class MemoryPackageSource:
    """Answers every request with the same packages and keeps the requests, building nothing.

    Args:
        packages: What every request resolves to.
    """

    packages: MemoryPackageSet = field(default_factory=MemoryPackageSet)
    requests: list[PackageRequest] = field(default_factory=list)

    def resolve(self, cfg: Config, root: Path, package_type: str,
                options: Mapping[str, tuple[str, ...]], report: Callable[[str], None],
                warn: Callable[[str], None], compiler: str) -> MemoryPackageSet:
        """Records the request and returns the packages this source holds."""
        self.requests.append(PackageRequest(package_type, dict(options), compiler))
        return self.packages

