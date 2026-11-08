# package_source_store.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Resolves the packages a build links through the package store of the configuration.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping
from pathlib import Path

from sushicore.provision.sushi_packages import PackageSet

from ..config import Config


@typing.final
class StorePackageSource:
    """Keeps the promise of :class:`PackageSource` through ``cfg.sushi_packages``."""

    def resolve(self, cfg: Config, root: Path, package_type: str,
                options: Mapping[str, tuple[str, ...]], report: Callable[[str], None],
                warn: Callable[[str], None], compiler: str) -> PackageSet:
        """Returns the built packages of ``cfg.sushi_packages``, building a missing one."""
        return cfg.sushi_packages(root, package_type, options, report=report, warn=warn,
                                  compiler=compiler)
