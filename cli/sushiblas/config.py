# config.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Loads the layered configuration of the SushiBLAS CLI.

Precedence, low to high: built-in defaults, config.toml, config.local.toml, SB_* environment
variables. See cli/sushiblas/README.md.
"""

from __future__ import annotations

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
    default_target="sushiblas_example",
    # SUSHIRUNTIME_DIR is unprefixed because every CLI that builds against the checkout reads it.
    extra_env_overrides={"sushiruntime_dir": "SUSHIRUNTIME_DIR"},
    # SYCL, MSVC's INCLUDE/LIB, and every SUSHI* checkout override.
    env_of_interest=("SYCL", "SUSHI", "INCLUDE", "LIB"),
)

_MODULE = ModuleConfig(PROFILE)

find_project_root = _MODULE.find_project_root
config_dir = _MODULE.config_dir

#: Maps each field to its environment variable; `sb config` needs the reverse map.
_ENV_OVERRIDES = PROFILE.env_overrides()


@dataclass
class Config(StackConfig):
    """Resolved, platform-specific tool configuration for the SushiBLAS build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths) from
    :class:`ToolConfig` and the shared-tree toolchain resolution from
    :class:`StackConfig`. ``cxx`` empty means "discover it": the bundled clang++
    under the shared toolchain if present, else a clang++ on PATH.
    """

    # Empty means "../sushiruntime" relative to the project root, as in cmake/Runtime.cmake.
    sushiruntime_dir: str = ""

    # Run defaults: sushiblas_example is a real target (examples/CMakeLists.txt).
    target_bin: str = "sushiblas_example"

    def runtime_dir(self, root: Path) -> Path:
        """Resolve the SushiRuntime checkout this build links against.

        @param root The SushiBLAS project root (the directory holding CMakeLists.txt).
        @return The runtime directory, defaulting to ``<root>/../sushiruntime``
                (which, inside a SushiStack workspace, is the sibling module).
        """
        return self.sibling_dir(root, "sushiruntime", self.sushiruntime_dir)


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform."""
    return _MODULE.load(Config)
