# project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs SushiBLAS's build, test, run, clean and docs through sushicore's CMake driver.

The compiler and vcpkg come from the config overrides, then the workspace `dependencies/`
tree, then the machine tree under `~/.sushisystems`. The suites are the CTest labels that
tests/CMakeLists.txt assigns from the GTest suite-name prefix. See cli/sushiblas/README.md.
"""

from __future__ import annotations

import enum
import shutil
from collections.abc import Sequence
from pathlib import Path

from sushicore.cmake_cache import cached_value
from sushicore.cmake_driver import CMakeDriver
from sushicore.proc import Runner
from sushicore.toolchain_args import c_compiler_for as _resolved_c_compiler
from sushicore.toolchain_args import vcpkg_prefix as _vcpkg_prefix

from .. import console
from ..config import Config, PROFILE, find_project_root, load_config
from ..env import load_build_env
from . import discovery


class BuildType(str, enum.Enum):
    """Names the build types `sb build --type` accepts."""

    release = "release"
    debug = "debug"
    relwithdebinfo = "relwithdebinfo"


class Suite(str, enum.Enum):
    """Names the test suites `sb test --suite` accepts."""

    # CTest labels that tests/CMakeLists.txt assigns from the GTest suite-name prefix.
    unit = "unit"
    regression = "regression"
    integration = "integration"
    functional = "functional"   # umbrella: unit + regression + integration
    all = "all"

    # Not a CTest label and not part of `all`; see :func:`test_package`.
    package = "package"


# Maps each CTest-backed suite to its `ctest -L` regex; see cli/sushiblas/README.md.
_SUITE_LABEL_REGEX = {
    Suite.unit: "^unit$",
    Suite.regression: "^regression$",
    Suite.integration: "^integration$",
    Suite.functional: "^(unit|integration|regression)$",
}

_CMAKE_BUILD_TYPE = {
    BuildType.release: "Release",
    BuildType.debug: "Debug",
    BuildType.relwithdebinfo: "RelWithDebInfo",
}


def _build_dir(root: Path) -> Path:
    """Returns the build tree of the checkout at *root*."""
    return root / "build"


_RUNNER = Runner(console, PROFILE.program, catch_interrupt=True)
_DRIVER = CMakeDriver(console, _RUNNER)


def _check_runtime(cfg: Config, root: Path) -> int:
    """Fail early (with guidance) when the SushiRuntime sibling is missing.

    SushiBLAS cannot configure without it (see cmake/Runtime.cmake) — the
    runtime provides the SYCL target and the add_sycl_to_target command the
    project's CMakeLists pulls in.
    """
    runtime = cfg.runtime_dir(root)
    if (runtime / "CMakeLists.txt").is_file():
        return 0
    console.error(
        f"SushiRuntime sibling not found at: {runtime}\n"
        f"  - clone it next to this repo, or\n"
        f"  - set sushiruntime_dir in cli/config.local.toml, or SUSHIRUNTIME_DIR in the env.")
    return 1


def _toolchain_args(cfg: Config, root: Path) -> list[str]:
    """The compiler and host-tool flags every configure in this tree must share.

    Used by the project's own configure and by the package consumer's (see
    :func:`test_package`). They are factored out rather than written twice
    because the consumer configure is precisely where a missing one costs an
    hour to diagnose: clang-cl's CMake platform module insists on a resource
    compiler, and vcpkg's toolchain file is what puts hwloc and GoogleTest
    where find_package can see them. A copy of this list would drift the moment
    one of the two configures gained a flag.

    CMAKE_PREFIX_PATH is deliberately absent: it is the one setting the two
    configures legitimately disagree on, since the consumer must also see the
    install prefix. Each caller sets its own.

    @param cfg The resolved configuration.
    @param root The project root.
    @return The shared -D flags, in the order cmake should receive them.
    """
    cxx = cfg.resolved_compiler(root)
    args = [
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_C_COMPILER={_resolved_c_compiler(cxx)}",
    ]
    if cfg.is_windows:
        if cfg.ninja_exe:
            args.append(f"-DCMAKE_MAKE_PROGRAM={cfg.expand(cfg.ninja_exe)}")
        if cfg.pkgconf_exe:
            args.append(f"-DPKG_CONFIG_EXECUTABLE={cfg.expand(cfg.pkgconf_exe)}")
        if cfg.rc_exe:
            args.append(f"-DCMAKE_RC_COMPILER={cfg.expand(cfg.rc_exe)}")

    vcpkg = cfg.resolved_vcpkg(root)
    if vcpkg:
        args += [
            f"-DCMAKE_TOOLCHAIN_FILE={vcpkg}/scripts/buildsystems/vcpkg.cmake",
            f"-DVCPKG_ROOT={vcpkg}",
        ]
        if cfg.is_windows:
            args.append(f"-DVCPKG_TARGET_TRIPLET={cfg.vcpkg_triplet}")

    return args


def _configure_args(cfg: Config, root: Path, build_dir: Path, build_type: str,
                    defines: Sequence[str] | None = None) -> list[str]:
    """Returns the cmake configure command line for this tree.

    Raises:
        ValueError: An entry of *defines* is not VAR=VALUE.
    """
    runtime = cfg.runtime_dir(root)

    args = [
        _DRIVER.cmake(cfg), "-S", str(root), "-B", str(build_dir), "-G", cfg.generator,
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DSUSHIRUNTIME_DIR={runtime}",
    ]
    args += _toolchain_args(cfg, root)
    if cfg.is_windows and (prefix := _vcpkg_prefix(cfg, root)):
        args.append(f"-DCMAKE_PREFIX_PATH={prefix}")

    # Caller-supplied cache variables come last, so no default set above outranks them.
    for entry in defines or ():
        if "=" not in entry:
            raise ValueError(
                f"-D expects VAR=VALUE, got {entry!r}. "
                "Use VAR= to clear a variable.")
        args.append(f"-D{entry}")

    return args


def build(build_type: BuildType, clean: bool = False,
          defines: Sequence[str] | None = None) -> int:
    """Configures and builds the project, replacing a tree it cannot reuse."""
    console.header("Project Build")
    root = find_project_root()
    cfg = load_config()
    cmake_build_type = _CMAKE_BUILD_TYPE[build_type]
    build_dir = _build_dir(root)

    if (rc := _check_runtime(cfg, root)) != 0:
        return rc

    if clean:
        _DRIVER.clean_tree(build_dir)

    console.info(f"Runtime: {cfg.runtime_dir(root)}")
    console.info(f"Compiler: {cfg.resolved_compiler(root)}")

    env = load_build_env(cfg, build_dir)

    fresh = _DRIVER.needs_configure(build_dir, cfg.generator, root=root,
                                    expect={"CMAKE_BUILD_TYPE": cmake_build_type})
    if fresh:
        if build_dir.is_dir():
            console.info(
                "build/ was configured for another build type, generator or source path; "
                "starting from an empty tree.")
            _DRIVER.clean_tree(build_dir)
        console.info(f"Configuring CMake... (type={build_type.value})")
    else:
        console.info("Reconfiguring in place...")
    try:
        configure_args = _configure_args(cfg, root, build_dir, cmake_build_type, defines)
    except ValueError as exc:
        console.error(str(exc))
        return 2

    rc = _DRIVER.configure(configure_args, cwd=root, env=env)
    if rc != 0:
        console.error("CMake configure failed.")
        return rc
    build_dir.mkdir(parents=True, exist_ok=True)

    console.info("Building...")
    rc = _DRIVER.compile(cfg, build_dir, cwd=root, env=env, config=cmake_build_type)
    if rc == 0:
        console.success("Build completed successfully!")
    else:
        console.error("Build failed.")
    return rc


def test(suite: Suite = Suite.functional, filter: str | None = None, repeat: int = 0) -> int:
    """Run the test suite via CTest labels.

    For GTest-level knobs (per-case filtering, shuffle, break-on-failure) run
    the binary directly: sb run sushiblas_tests -- --gtest_filter=Gemm.* --gtest_shuffle

    ``package`` is the one suite that is not a CTest label; it delegates to
    :func:`test_package`, which builds a separate project.
    """
    console.header("Project Test")
    root = find_project_root()
    cfg = load_config()
    build_dir = _build_dir(root)
    if not build_dir.is_dir():
        console.error("build/ not found. Run `sb build` first.")
        return 1

    env = load_build_env(cfg, build_dir)

    if suite is Suite.package:
        if filter or repeat:
            # Both are ctest knobs and this suite runs no ctest, so they are refused.
            console.error("--filter and --repeat do not apply to the package suite.")
            return 2
        return test_package(cfg, root, build_dir, env)

    if suite is Suite.all:
        # `all` covers every CTest test, which leaves the package suite out.
        console.info(
            "'all' covers every CTest test. The package suite is not among them "
            "(it builds a separate project): run `sb test --suite package`.")

    return _DRIVER.ctest_run(
        cfg, build_dir, env,
        label_regex=None if suite is Suite.all else _SUITE_LABEL_REGEX[suite],
        filter=filter, repeat=repeat)


def _deploy_consumer_dlls(cfg: Config, root: Path, prefix: Path, dest: Path) -> None:
    """Put the shared libraries the consumer needs beside its executable.

    In-tree targets get these from a POST_BUILD copy (see tests/CMakeLists.txt);
    a real out-of-tree consumer would deploy them itself, and this function is
    this CLI standing in for that consumer. Without them the binary exits with
    0xC0000135 before reaching main, which looks like a package failure and is
    not one.

    @param cfg The resolved configuration.
    @param root The project root.
    @param prefix The install prefix the package was installed to.
    @param dest The directory holding the consumer executable.
    """
    if not cfg.is_windows:
        return  # ELF consumers resolve via RPATH; nothing to stage.

    sources = list((prefix / "bin").glob("*.dll"))
    if vcpkg_prefix := _vcpkg_prefix(cfg, root):
        sources += list(Path(vcpkg_prefix, "bin").glob("hwloc*.dll"))

    for dll in sources:
        shutil.copy2(dll, dest / dll.name)
    console.info(f"Deployed {len(sources)} shared libraries beside the consumer.")


def test_package(cfg: Config, root: Path, build_dir: Path, env: dict[str, str]) -> int:
    """Install the package and build tests/package/ against it, as a consumer would.

    This is the only test in the repo that exercises what ``find_package(SushiBLAS)``
    actually hands a third party, and it is the reason it cannot be a CTest label:
    a package that works only when the source tree happens to be next to it is not
    a package, so the consumer must configure from an install prefix with no access
    to this tree. It also compiles the public headers at C++20 while the library
    itself is built at C++17 - see tests/package/CMakeLists.txt for why that
    difference is the point rather than an oversight.

    Both intermediates live under build/, so `sb clean` reclaims them and nothing
    lands in the source tree.

    @param cfg The resolved configuration.
    @param root The project root.
    @param build_dir The project's own build tree, which must already be built.
    @param env The snapshotted build environment.
    @return 0 when the consumer builds and runs, non-zero otherwise.
    """
    build_type = cached_value(build_dir, "CMAKE_BUILD_TYPE") or "Release"
    prefix = build_dir / "package" / "prefix"
    consumer = build_dir / "package" / "consumer"

    # The install starts from an empty prefix, so a deleted header cannot satisfy an include.
    if prefix.is_dir():
        shutil.rmtree(prefix, ignore_errors=True)

    console.info(f"Installing to {prefix}...")
    rc = _RUNNER.run([_DRIVER.cmake(cfg), "--install", str(build_dir),
                      "--prefix", str(prefix), "--config", build_type], root, env)
    if rc != 0:
        console.error("Install failed; the package could not be staged.")
        return rc

    prefix_path = str(prefix)
    if vcpkg_prefix := _vcpkg_prefix(cfg, root):
        prefix_path = f"{prefix_path};{vcpkg_prefix}"

    console.info("Configuring the out-of-tree consumer...")
    configure = [
        _DRIVER.cmake(cfg), "-S", str(root / "tests" / "package"), "-B", str(consumer),
        "-G", cfg.generator,
        # The CXX-only consumer leaves some shared flags unread; only that warning is silenced.
        "--no-warn-unused-cli",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_PREFIX_PATH={prefix_path}",
    ]
    configure += _toolchain_args(cfg, root)
    rc = _DRIVER.configure(configure, cwd=root, env=env)
    if rc != 0:
        console.error(
            "The consumer could not configure against the installed package.\n"
            "  - a FATAL_ERROR above about leaked flags means the exported "
            "interface carries development-only options; see "
            "cmake/SushiBLASBuildPolicy.cmake.")
        return rc

    console.info("Building the consumer...")
    rc = _DRIVER.compile(cfg, consumer, cwd=root, env=env, config=build_type)
    if rc != 0:
        console.error(
            "The consumer failed to compile against the installed headers.\n"
            "  - a missing header means it is not covered by install(DIRECTORY) "
            "in src/CMakeLists.txt.\n"
            "  - the consumer compiles at C++20 while the library builds at "
            "C++17, so a construct valid only in one of the two fails here.")
        return rc

    exe = discovery.match_by_name(consumer, "sushiblas_package_consumer")
    if exe is None:
        console.error("Built the consumer but could not find its executable.")
        return 1

    _deploy_consumer_dlls(cfg, root, prefix, exe.parent)

    console.info(f"Executing: {exe.name}")
    rc = _RUNNER.run([str(exe)], consumer, env)
    if rc == 0:
        console.success("Package consumer built and ran against the install prefix.")
    else:
        console.error("The package consumer failed at runtime.")
    return rc


def run(target: str | None = None, sort: bool = False,
        app_args: list[str] | None = None) -> int:
    """Runs a built executable with *app_args* and returns its exit code."""
    console.header("Project Run")
    root = find_project_root()
    cfg = load_config()
    build_dir = _build_dir(root)
    if not build_dir.is_dir():
        console.error("build/ not found. Run `sb build` first.")
        return 1

    env = load_build_env(cfg, build_dir)

    exe = discovery.resolve(build_dir, console, target=target, sort=sort, default=cfg.target_bin)
    if exe is None:
        return 1

    console.info(f"Executing: {exe.name}")
    console.console.print(f"[muted]{exe}[/muted]")
    return _RUNNER.run([str(exe), *(app_args or [])], root, env)


def clean() -> int:
    """Removes the build tree."""
    console.header("Project Clean")
    _DRIVER.clean_tree(_build_dir(find_project_root()))
    return 0


def docs() -> int:
    """Generates the Doxygen API reference from the repository's Doxyfile."""
    console.header("Doxygen Generation")
    root = find_project_root()
    cfg = load_config()
    env = load_build_env(cfg, _build_dir(root))
    doxyfile = root / "Doxyfile"
    (root / "docs").mkdir(exist_ok=True)
    return _DRIVER.doxygen(cfg, doxyfile, root, env)
