"""Layered configuration loading for the SushiBLAS CLI.

Precedence (low to high):

    built-in defaults -> config.toml -> config.local.toml -> SB_* env vars

SushiBLAS is a head of the stack: it consumes the shared SushiStack SYCL
toolchain and vcpkg tree rather than provisioning its own, and it builds against
a live SushiRuntime sibling checkout (D3 layered CMake resolution) rather than a
vendored copy. So this module selects no toolchain -- resolving the compiler and
vcpkg out of the shared tree is :class:`sushicore.stack_config.StackConfig`'s
job, and locating the project and layering the files is
:class:`sushicore.module_config.ModuleConfig`'s. What is left here is what is
actually SushiBLAS's: its identity, and the one sibling it builds against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sushicore.module_config import ModuleConfig
from sushicore.profile import ModuleProfile
from sushicore.stack_config import StackConfig

#: How SushiBLAS describes itself to the shared machinery. Stated once here and
#: read by config loading, the console, executable discovery and diagnostics --
#: which is the point: they cannot disagree about what this module is.
PROFILE = ModuleProfile(
    name="SushiBLAS",
    program="sb",
    env_prefix="SB",
    root_marker="CMakeLists.txt",
    siblings=("sushiruntime",),
    default_target="sushiblas_example",
    # Not prefixed: SUSHIRUNTIME_DIR names the same checkout for every CLI that
    # builds against it, so it must read the same variable they do.
    extra_env_overrides={"sushiruntime_dir": "SUSHIRUNTIME_DIR"},
    # SYCL, MSVC's INCLUDE/LIB, and every SUSHI* checkout override.
    env_of_interest=("SYCL", "SUSHI", "INCLUDE", "LIB"),
)

_MODULE = ModuleConfig(PROFILE)

find_project_root = _MODULE.find_project_root
config_dir = _MODULE.config_dir

#: Field -> environment variable, derived from the profile's prefix. Exposed as
#: a module attribute because `sb config` reports which values came from the
#: environment and needs the reverse map.
_ENV_OVERRIDES = PROFILE.env_overrides()


@dataclass
class Config(StackConfig):
    """Resolved, platform-specific tool configuration for the SushiBLAS build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths) from
    :class:`ToolConfig` and the shared-tree toolchain resolution from
    :class:`StackConfig`. ``cxx`` empty means "discover it": the bundled clang++
    under the shared toolchain if present, else a clang++ on PATH.
    """

    # The SushiRuntime sibling checkout. Empty means "../sushiruntime" relative
    # to the project root, matching SUSHIRUNTIME_DIR's default in
    # cmake/Runtime.cmake.
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

    def standalone_deps_dir(self, root: Path) -> Path:
        """Outside a workspace, fall back to the runtime sibling's own bundle.

        SushiRuntime is the ultimate owner of the toolchain bundle, so a
        standalone SushiBLAS checkout borrows it from the runtime it builds
        against rather than provisioning a second copy.
        """
        return self.runtime_dir(root) / "dependencies"


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform."""
    return _MODULE.load(Config)
