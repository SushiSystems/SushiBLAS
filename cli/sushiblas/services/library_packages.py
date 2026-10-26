# library_packages.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Takes the Sushi libraries `sb` links as built packages and puts their run-time folders on the path.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sushicore.provision.sushi_packages import PackageSet, RunDirs
from sushicore.provision.sushi_packages import configure_arguments  # noqa: F401  (callers take it from here)

from ..config import Config

#: The file of a build tree that names the run-time folders of the packages it linked.
K_RUN_DIRS_FILE = ".sushiblas_package_dirs.json"

#: The library whose package is built with the distributed layer when SushiBLAS is.
K_RUNTIME = "sushiruntime"
K_DISTRIBUTED_FLAG = "--distributed"


def package_options(distributed: bool) -> Mapping[str, tuple[str, ...]]:
    """Return the extra `build` arguments each library takes for this SushiBLAS build.

    The cluster module needs the runtime's distributed layer, so ``--distributed`` of SushiBLAS
    becomes ``--distributed`` of the SushiRuntime package.
    """
    return {K_RUNTIME: (K_DISTRIBUTED_FLAG,)} if distributed else {}


def resolve(cfg: Config, root: Path, build_type: str, compiler: str, distributed: bool,
            report: Callable[[str], None], warn: Callable[[str], None]) -> PackageSet:
    """Return the built packages of the closure of ``depends_on``, building a missing one.

    The closure is SushiLog, SushiProfiler and SushiRuntime, the three libraries
    ``cli/sushistack.deps.toml`` names.

    Args:
        cfg: The resolved configuration.
        root: The project root.
        build_type: The build type of this configure, such as ``release``.
        compiler: The compiler of this build; the packages are built with it.
        distributed: Whether the cluster module is built.
        report: Receives one ``reused`` or ``built`` line per library.
        warn: Receives the warnings of the resolver.

    Raises:
        sushicore.provision.sushi_packages.PackageError: A package cannot be chosen.
    """
    return cfg.sushi_packages(root, build_type, package_options(distributed),
                              report=report, warn=warn, compiler=compiler)


#: The run-time folders of the packages a tree linked, kept in the tree for the commands that run it.
_RUN_DIRS = RunDirs(K_RUN_DIRS_FILE)

def record_run_dirs(tree: Path, dirs: Sequence[str]) -> None:
    """Write the run-time folders of the packages *tree* linked, for the commands that run it."""
    _RUN_DIRS.save(tree, dirs)


def read_run_dirs(tree: Path) -> list[str]:
    """Return the folders :func:`record_run_dirs` wrote for *tree*, none when it wrote none."""
    return _RUN_DIRS.read(tree)


def prepend_run_dirs(cfg: Config, env: dict[str, str], dirs: Sequence[str]) -> None:
    """Put *dirs* first on the search path of *env*, where a binary finds the package libraries."""
    _RUN_DIRS.prepend(env, dirs, cfg.is_windows)
