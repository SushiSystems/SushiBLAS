"""Layered configuration loading for the SushiBLAS CLI.

Precedence (lowest to highest):
    built-in defaults -> config.toml -> config.local.toml -> SB_* env vars

The active platform's ``[tool.<platform>]`` table is merged over the common
``[tool]`` table, so a single file describes both Linux and Windows.

Unlike the engine, SushiBLAS does not yet consume the shared SushiStack SYCL
toolchain: today's ``CMakeLists.txt`` calls ``find_package(IntelSYCL REQUIRED)``
against a real Intel oneAPI DPC++ install, and the SushiRuntime dependency is a
vendored copy checked into ``lib/``/``include/SushiRuntime`` rather than a
sibling checkout resolved by CMake. Both of those are being replaced (SB-2's
backend rework, SB-0's live runtime dependency) but haven't landed yet, so this
config models the *current* build, not the target one — see the ST-1 plan's
scope note. When SB-0/SB-2 land, only this file and env.py need to change.
"""

from __future__ import annotations

import os
import platform

from dataclasses import dataclass
from pathlib import Path

# Domain-agnostic config plumbing shared by every Sushi* CLI. The generic build-
# tool schema (cmake/ninja/vcpkg paths) and the layered-load skeleton live in
# sushicli; this repo adds only the BLAS-specific fields below.
from sushicli.config_base import ToolConfig, load_tool_config
from sushicli.workspace import has_marker, resolve_env_path, walk_up


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repo root by walking up from CWD until CMakeLists.txt is found.

    The CLI is installed (pip/pipx) outside the repo, so the package location
    tells us nothing about where the project lives — the invocation directory
    does. Run any `sb` command from anywhere inside the checkout.
    """
    root = walk_up(start or Path.cwd(), has_marker("CMakeLists.txt"))
    if root is None:
        raise SystemExit(
            "Not inside a SushiBLAS project: no CMakeLists.txt found in the "
            "current directory or any parent. cd into the repo and try again."
        )
    return root


def config_dir(root: Path | None = None) -> Path:
    """Directory holding config.toml / config.local.toml (the repo's cli/)."""
    root = root or find_project_root()
    return root / "cli"


# Maps a Config field to the SB_* env var that overrides it.
_ENV_OVERRIDES = {
    "cxx": "SB_CXX",
    "generator": "SB_CMAKE_GENERATOR",
    "vcpkg_root": "SB_VCPKG_ROOT",
    "vs_vcvars": "SB_VCVARS",
    "ninja_exe": "SB_NINJA",
    "cmake_exe": "SB_CMAKE",
    "ctest_exe": "SB_CTEST",
    "rc_exe": "SB_RC",
    "pkgconf_exe": "SB_PKGCONF",
    "doxygen_exe": "SB_DOXYGEN",
    "vcpkg_triplet": "SB_VCPKG_TRIPLET",
    "oneapi_root": "SB_ONEAPI_ROOT",
    "target_bin": "SB_TARGET_BIN",
}


@dataclass
class Config(ToolConfig):
    """Resolved, platform-specific tool configuration for the SushiBLAS build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths, etc.)
    from :class:`ToolConfig`. SushiBLAS additionally needs a real Intel oneAPI
    DPC++ install (``cxx`` empty means "derive icx-cl/icpx from oneapi_root").
    """

    # The Intel oneAPI root (e.g. "C:/Program Files (x86)/Intel/oneAPI"). Empty
    # means "read the ONEAPI_ROOT environment variable" (matching setvars.bat's
    # own convention and today's scripts/win/config.ini).
    oneapi_root: str = ""

    # Run defaults: sushiblas_example is a real target (examples/CMakeLists.txt),
    # unlike the stale "sushiblas_app" the old scripts/ pointed at.
    target_bin: str = "sushiblas_example"

    def resolved_oneapi_root(self) -> str:
        """The oneAPI root: explicit config value, else $ONEAPI_ROOT, else ''."""
        if self.oneapi_root:
            return self.expand(self.oneapi_root)
        return os.environ.get("ONEAPI_ROOT", "")

    def resolved_compiler(self) -> str:
        """The C++ compiler driving the build: explicit, then derived, then PATH.

        @return A configured cxx, else icx-cl/icpx under the resolved oneAPI
                root's compiler/latest/bin, else the bare exe name for PATH
                resolution (mirroring CMakeLists.txt's own oneAPI discovery).
        """
        if self.cxx:
            return self.expand(self.cxx)
        exe = "icx-cl.exe" if self.is_windows else "icpx"
        root = self.resolved_oneapi_root()
        if root:
            return str(Path(root) / "compiler" / "latest" / "bin" / exe)
        return exe

    def sushistack_home(self, root: Path) -> Path | None:
        """Locate the SushiStack workspace root, or None when standalone.

        SushiStack is the umbrella that provisions one shared dependency tree for
        every module. Resolution: ``SUSHISTACK_HOME`` env var, then a walk up from
        the project root looking for the ``.sushistack`` marker.

        @param root The SushiBLAS project root.
        @return The workspace root, or None when not inside a workspace.
        """
        home = resolve_env_path("SUSHISTACK_HOME")
        if home:
            return home
        return walk_up(root, has_marker(".sushistack"))

    def deps_dir(self, root: Path) -> Path:
        """The shared dependency tree ``ss install`` provisions vcpkg into.

        @param root The SushiBLAS project root.
        @return The workspace's ``dependencies`` tree, or ``<root>/dependencies``
                when standalone (not inside a SushiStack workspace).
        """
        override = os.environ.get("SUSHISTACK_DEPS_DIR")
        if override:
            return Path(self.expand(override))
        home = self.sushistack_home(root)
        if home:
            return home / "dependencies"
        return root / "dependencies"

    def resolved_vcpkg(self, root: Path) -> str:
        """The vcpkg root: explicit, then the shared bundled tree, else ''.

        @param root The SushiBLAS project root.
        @return An absolute vcpkg root, or '' when none is configured or bundled.
        """
        if self.vcpkg_root:
            return self.expand(self.vcpkg_root)
        bundled = self.deps_dir(root) / "vcpkg"
        return str(bundled) if bundled.is_dir() else ""


def _shared_config_local() -> Path | None:
    """The workspace-shared config.local.toml ``ss install`` writes, if any.

    Inside a SushiStack workspace the machine-specific tool paths (compiler,
    vcpkg, cmake) are resolved once by ``ss`` and written to ``<home>/cli/
    config.local.toml``; every module reads them from there so nothing is
    configured twice.
    """
    home = resolve_env_path("SUSHISTACK_HOME")
    if home is None:
        try:
            home = walk_up(find_project_root(), has_marker(".sushistack"))
        except SystemExit:
            home = None
    return (home / "cli" / "config.local.toml") if home else None


def load_config() -> Config:
    """Load and resolve the layered configuration for the current platform.

    Precedence (low to high): repo config.toml -> workspace-shared
    config.local.toml (written by ``ss``) -> repo config.local.toml -> env.
    """
    plat = platform.system().lower()  # 'windows' | 'linux' | 'darwin'

    cfg_dir = config_dir()
    shared = _shared_config_local()
    sources = [cfg_dir / "config.toml"]
    if shared is not None:
        sources.append(shared)
    sources.append(cfg_dir / "config.local.toml")
    return load_tool_config(Config, sources, plat, _ENV_OVERRIDES)
