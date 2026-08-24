"""Layered configuration loading for the SushiBLAS CLI.

Precedence (lowest to highest):
    built-in defaults -> config.toml -> config.local.toml -> SB_* env vars

The active platform's ``[tool.<platform>]`` table is merged over the common
``[tool]`` table, so a single file describes both Linux and Windows.

SushiBLAS is a head of the stack: it consumes the shared SushiStack SYCL
toolchain and vcpkg tree rather than provisioning its own, and it builds
against a live SushiRuntime sibling checkout (D3 layered CMake resolution)
rather than a vendored copy. So this config does not select a toolchain — it
locates the shared ``<workspace>/dependencies`` tree (provisioned by
``ss install``) for the compiler and vcpkg, the runtime sibling module it
builds against, plus the handful of host tools (cmake, ninja, vcvars) a build
needs. Standalone (outside a workspace) it falls back to the runtime's own
bundled dependencies for backward compatibility. Mirrors sushiengine's `se`
CLI exactly (same toolchain-bundle contract, one sibling module).
"""

from __future__ import annotations

import os
import platform

from dataclasses import dataclass
from pathlib import Path

# Domain-agnostic config plumbing shared by every Sushi* CLI. The generic build-
# tool schema (cmake/ninja/vcpkg paths) and the layered-load skeleton live in
# sushicore; this repo adds only the BLAS-specific fields below.
from sushicore.config_base import ToolConfig, load_tool_config
from sushicore.workspace import has_marker, resolve_env_path, walk_up


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
    "sushiruntime_dir": "SUSHIRUNTIME_DIR",
    "target_bin": "SB_TARGET_BIN",
}


@dataclass
class Config(ToolConfig):
    """Resolved, platform-specific tool configuration for the SushiBLAS build.

    Inherits the generic host build-tool fields (cmake/ninja/vcpkg paths, etc.)
    from :class:`ToolConfig`. SushiBLAS selects no SYCL toolchain of its own —
    it consumes the shared one — so it adds only the sibling-runtime checkout
    below. ``cxx`` (from ToolConfig) empty means "discover it": the runtime's
    bundled clang++ (under <runtime>/dependencies/toolchains/llvm-sycl/bin) if
    present, else a clang++ on PATH.
    """

    # The SushiRuntime sibling checkout. Empty means "../sushiruntime relative
    # to the project root" (matching SUSHIRUNTIME_DIR's default in
    # cmake/Runtime.cmake).
    sushiruntime_dir: str = ""

    # Run defaults: sushiblas_example is a real target (examples/CMakeLists.txt).
    target_bin: str = "sushiblas_example"

    def runtime_dir(self, root: Path) -> Path:
        """Resolve the SushiRuntime checkout this build links against.

        @param root The SushiBLAS project root (the directory holding CMakeLists.txt).
        @return The runtime directory, defaulting to ``<root>/../sushiruntime``
                (which, inside a SushiStack workspace, is the sibling module).
        """
        if self.sushiruntime_dir:
            return Path(self.expand(self.sushiruntime_dir)).resolve()
        return (root / ".." / "sushiruntime").resolve()

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
        """The shared dependency tree this build resolves its toolchain from.

        Inside a SushiStack workspace that is ``<workspace>/dependencies`` (shared
        by every module); standalone it falls back to the runtime sibling's own
        bundled ``dependencies`` tree. ``SUSHISTACK_DEPS_DIR`` overrides both.

        @param root The SushiBLAS project root.
        @return The dependency tree root.
        """
        override = os.environ.get("SUSHISTACK_DEPS_DIR")
        if override:
            return Path(self.expand(override))
        home = self.sushistack_home(root)
        if home:
            return home / "dependencies"
        return self.runtime_dir(root) / "dependencies"

    def bundled_clang(self, root: Path) -> str:
        """The shared bundled clang++ path, or '' when the bundle is absent.

        @param root The SushiBLAS project root.
        @return The absolute clang++ path under the shared toolchain bundle, or an
                empty string when it has not been installed there.
        """
        exe = "clang++.exe" if self.is_windows else "clang++"
        candidate = self.deps_dir(root) / "toolchains" / "llvm-sycl" / "bin" / exe
        return str(candidate) if candidate.is_file() else ""

    def resolved_compiler(self, root: Path) -> str:
        """The C++ compiler to drive the build: explicit, then bundled, then PATH.

        @param root The SushiBLAS project root.
        @return A configured cxx, else the shared bundled clang++, else bare
                'clang++' for PATH resolution.
        """
        if self.cxx:
            return self.expand(self.cxx)
        bundled = self.bundled_clang(root)
        return bundled or "clang++"

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
