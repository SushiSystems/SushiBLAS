# library_packages.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Takes the Sushi libraries `sb` links as built packages and puts their run-time folders on the path.

See cli/sushiblas/README.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from sushicore.build_tree import package_build_type
from sushicore.provision.sushi_packages import RunDirs

from ..config import Config
from .sanitizer import Sanitizer

#: The file of a build tree that names the run-time folders of the packages it linked.
K_RUN_DIRS_FILE = ".sushiblas_package_dirs.json"

#: The library whose package is built with the distributed layer when SushiBLAS is.
K_RUNTIME = "sushiruntime"
K_DISTRIBUTED_FLAG = "--distributed"


#: The libraries whose CLI takes the sanitizer as ``--sanitize thread``.
K_SANITIZE_FLAG_LIBRARIES = ("sushilog", "sushiprofiler")

#: The package build type of a ThreadSanitizer tree; TSan needs the symbols to name a race.
K_THREAD_PACKAGE_TYPE = "relwithdebinfo"


def package_options(distributed: bool,
                    sanitizer: Sanitizer | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return the extra `build` arguments each library takes for this SushiBLAS build.

    The cluster module needs the runtime's distributed layer, so ``--distributed`` of SushiBLAS
    becomes ``--distributed`` of the SushiRuntime package. A ThreadSanitizer tree needs the three
    packages built with it, because TSan cannot see a synchronisation edge that runs through an
    uninstrumented library: SushiLog and SushiProfiler take ``--sanitize thread`` and
    SushiRuntime takes ``--type tsan``, which follows the ``--type`` the builder already passed.
    """
    options: dict[str, tuple[str, ...]] = {}
    if distributed:
        options[K_RUNTIME] = (K_DISTRIBUTED_FLAG,)
    if sanitizer is Sanitizer.THREAD:
        for name in K_SANITIZE_FLAG_LIBRARIES:
            options[name] = ("--sanitize", "thread")
        options[K_RUNTIME] = options.get(K_RUNTIME, ()) + ("--type", "tsan")
    return options


def package_type(build_type: str, sanitizer: Sanitizer | None) -> str:
    """Return the build type the packages of a tree are built as.

    A ThreadSanitizer tree needs the symbols to name a race, so it takes
    :data:`K_THREAD_PACKAGE_TYPE`; any other tree takes
    :func:`sushicore.build_tree.package_build_type` of its type and whether it is sanitized.

    Args:
        build_type: The build type of this configure, such as ``release``.
        sanitizer: The sanitizer set of this tree.
    """
    if sanitizer is Sanitizer.THREAD:
        return K_THREAD_PACKAGE_TYPE
    return package_build_type(build_type, sanitizer not in (None, Sanitizer.NONE))


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
