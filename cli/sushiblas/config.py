# config.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the layered configuration of the SushiBLAS CLI.

Precedence, low to high: built-in defaults, config.toml, config.local.toml, SB_* environment
variables. See cli/sushiblas/README.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sushicore.module_config import ModuleConfig
from sushicore.profile import ModuleProfile
from sushicore.stack_config import StackConfig

#: Describes SushiBLAS to the shared machinery; every consumer reads this one profile.
PROFILE = ModuleProfile(
    name="SushiBLAS",
    program="sb",
    env_prefix="SB",
    root_marker="CMakeLists.txt",
    siblings=("sushiruntime",),
    default_target="quickstart_demo",
    # SYCL, MSVC's INCLUDE/LIB, and every SUSHI* checkout override.
    env_of_interest=("SYCL", "SUSHI", "INCLUDE", "LIB"),
)

#: The compiler a consumer's package build hands the CLI; it outranks the configured one.
K_STACK_CXX_ENV = "SUSHISTACK_CXX"

_MODULE = ModuleConfig(PROFILE)

find_project_root = _MODULE.find_project_root
config_dir = _MODULE.config_dir

#: Maps each field to its environment variable; `sb config` reads it for the reverse map.
ENV_OVERRIDES = PROFILE.env_overrides()

#: Names ENV_OVERRIDES under the attribute sushicore.diag.Diagnostics reads.
_ENV_OVERRIDES = ENV_OVERRIDES


@dataclass
class Config(StackConfig):
    """Resolved, platform-specific tool configuration for the SushiBLAS build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths) from
    :class:`ToolConfig` and the shared-tree toolchain resolution from
    :class:`StackConfig`. ``cxx`` empty means "discover it": the bundled clang++
    under the shared toolchain if present, else a clang++ on PATH.
    """

    target_bin: str = PROFILE.default_target

    # The compiler SUSHISTACK_CXX names; empty outside a package build. It outranks `cxx`.
    stack_cxx: str = ""

    def resolved_compiler(self, root: Path) -> str:
        """Return the C++ compiler: SUSHISTACK_CXX, else the one the shared resolution finds."""
        if self.stack_cxx:
            return self.expand(self.stack_cxx)
        return super().resolved_compiler(root)


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform."""
    cfg = _MODULE.load(Config)
    cfg.stack_cxx = os.environ.get(K_STACK_CXX_ENV, "")
    return cfg
