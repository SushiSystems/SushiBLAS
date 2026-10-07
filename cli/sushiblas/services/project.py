# project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs SushiBLAS's build, test, run, clean and docs through sushicore's CMake driver."""

from __future__ import annotations

import shutil
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sushicore.build_env import StackBuildEnv
from sushicore.cmake_cache import cached_value
from sushicore.cmake_driver import CMakeDriver
from sushicore.discovery import ExecutableIndex
from sushicore.proc import Runner
from sushicore.toolchain_args import c_compiler_for
from sushicore.toolchain_args import vcpkg_prefix

from ..config import Config
from .build_type import K_CMAKE_BUILD_TYPE, BuildType
from .device import K_DEVICE_VARIABLE, device_selector, with_device
from .errors import (
    BuildTreeMissingError,
    ExecutableNotFoundError,
    InvalidDefineError,
    RuntimeMissingError,
    StepFailedError,
    SuiteOptionError,
    TreeNotRemovedError,
)
from .suite import K_SUITE_LABEL_REGEX, Suite

K_PACKAGE_CONSUMER = "sushiblas_package_consumer"
K_BENCHMARKS_VARIABLE = "SB_BUILD_BENCHMARKS"


@dataclass(frozen=True, slots=True)
class CleanResult:
    """Describes what `clean` did to the build tree.

    Args:
        tree: The build tree the command addressed.
        removed: True when the tree existed and is now gone; False when there was none.
    """

    tree: Path
    removed: bool


def validate_defines(defines: Sequence[str] | None) -> list[str]:
    """Returns the -D flags for *defines*.

    Raises:
        InvalidDefineError: An entry is not VAR=VALUE.
    """
    flags = []
    for entry in defines or ():
        if "=" not in entry:
            raise InvalidDefineError(entry)
        flags.append(f"-D{entry}")
    return flags


def benchmarks_flags(benchmarks: bool | None) -> list[str]:
    """Returns the -D flag that switches the benchmark programs on or off.

    Args:
        benchmarks: True builds them, False stops building them, None leaves the cached value.
    """
    if benchmarks is None:
        return []
    return [f"-D{K_BENCHMARKS_VARIABLE}={'ON' if benchmarks else 'OFF'}"]


@typing.final
class ProjectService:
    """Builds, tests, runs, cleans and documents one SushiBLAS checkout.

    Failures raise a :class:`ProjectError`; finished steps return the child's exit code.
    The console receives progress lines only.
    """

    def __init__(
        self,
        console,
        runner: Runner,
        driver: CMakeDriver,
        index: ExecutableIndex,
        build_env: StackBuildEnv,
        find_root: Callable[[], Path],
        load_config: Callable[[], Config],
    ) -> None:
        self._console = console
        self._runner = runner
        self._driver = driver
        self._index = index
        self._build_env = build_env
        self._find_root = find_root
        self._load_config = load_config

    def build(
        self,
        build_type: BuildType,
        clean: bool = False,
        defines: Sequence[str] | None = None,
        benchmarks: bool | None = None,
    ) -> int:
        """Configures and builds the project, replacing a tree it cannot reuse.

        *benchmarks* switches the benchmark programs on or off; None keeps the cached value.

        Raises:
            InvalidDefineError: An entry of *defines* is not VAR=VALUE; nothing was touched.
            RuntimeMissingError: The SushiRuntime sibling is absent.
            TreeNotRemovedError: A tree that had to go is still there.
            StepFailedError: Configure or compile failed.
        """
        define_flags = benchmarks_flags(benchmarks) + validate_defines(defines)
        root = self._find_root()
        cfg = self._load_config()
        cmake_build_type = K_CMAKE_BUILD_TYPE[build_type]
        build_dir = _build_dir(root)
        _check_runtime(cfg, root)

        if clean:
            self._remove_tree(build_dir)

        self._console.info(f"Runtime: {cfg.runtime_dir(root)}")
        self._console.info(f"Compiler: {cfg.resolved_compiler(root)}")
        env = self._build_env.load(cfg, build_dir)

        fresh = self._driver.needs_configure(
            build_dir,
            cfg.generator,
            root=root,
            expect={"CMAKE_BUILD_TYPE": cmake_build_type},
        )
        if fresh:
            if build_dir.is_dir():
                self._console.info(
                    "build/ was configured for another build type, generator or source path; "
                    "starting from an empty tree.",
                )
                self._remove_tree(build_dir)
            self._console.info(f"Configuring CMake... (type={build_type.value})")
        else:
            self._console.info("Reconfiguring in place...")

        configure_args = self._configure_args(cfg, root, build_dir, cmake_build_type)
        rc = self._driver.configure(configure_args + define_flags, cwd=root, env=env)
        if rc != 0:
            raise StepFailedError("CMake configure failed.", rc)
        build_dir.mkdir(parents=True, exist_ok=True)

        self._console.info("Building...")
        rc = self._driver.compile(cfg, build_dir, cwd=root, env=env, config=cmake_build_type)
        if rc != 0:
            raise StepFailedError("Build failed.", rc)
        return rc

    def test(self, suite: Suite = Suite.FUNCTIONAL, filter: str | None = None,
             repeat: int = 0, device: str | None = None) -> int:
        """Runs the CTest label group of *suite*, or the package consumer, and returns ctest's code.

        *device* names the SYCL device the tests run on; None leaves the environment's choice.

        Raises:
            InvalidDeviceError: *device* is not a value --device accepts; nothing was run.
            BuildTreeMissingError: There is no build tree.
            SuiteOptionError: *filter* or *repeat* was given with the package suite.
            StepFailedError: A step of the package suite failed.
        """
        _check_device(device)
        root = self._find_root()
        cfg = self._load_config()
        build_dir = _existing_build_dir(root)

        if suite is Suite.PACKAGE:
            if filter or repeat:
                raise SuiteOptionError()
            env = self._device_env(self._build_env.load(cfg, build_dir), device)
            return self._test_package(cfg, root, build_dir, env)

        env = self._device_env(self._build_env.load(cfg, build_dir), device)
        if suite is Suite.ALL:
            self._console.info(
                "'all' covers every CTest test. The package suite is not among them "
                "(it builds a separate project): run `sb test --suite package`.",
            )

        return self._driver.ctest_run(
            cfg,
            build_dir,
            env,
            label_regex=None if suite is Suite.ALL else K_SUITE_LABEL_REGEX[suite],
            filter=filter,
            repeat=repeat,
        )

    def run(self, target: str | None = None, sort: bool = False,
            app_args: Sequence[str] = (), device: str | None = None) -> int:
        """Runs a built executable with *app_args* and returns its exit code.

        *device* names the SYCL device the program runs on; None leaves the environment's choice.

        Raises:
            InvalidDeviceError: *device* is not a value --device accepts; nothing was run.
            BuildTreeMissingError: There is no build tree.
            ExecutableNotFoundError: *target*, or the configured default, matches nothing.
        """
        _check_device(device)
        root = self._find_root()
        cfg = self._load_config()
        build_dir = _existing_build_dir(root)
        env = self._device_env(self._build_env.load(cfg, build_dir), device)

        exe = self._pick_executable(build_dir, target, sort, cfg.target_bin)
        if exe is None:
            return 1

        self._console.info(f"Executing: {exe.name}")
        self._console.console.print(f"[muted]{exe}[/muted]")
        return self._runner.run([str(exe), *app_args], root, env)

    def clean(self) -> CleanResult:
        """Removes the build tree.

        Raises:
            TreeNotRemovedError: The tree is still there after the removal.
        """
        build_dir = _build_dir(self._find_root())
        return CleanResult(build_dir, self._remove_tree(build_dir))

    def docs(self) -> int:
        """Generates the Doxygen API reference from the repository's Doxyfile."""
        root = self._find_root()
        cfg = self._load_config()
        env = self._build_env.load(cfg, _build_dir(root))
        (root / "docs").mkdir(exist_ok=True)
        return self._driver.doxygen(cfg, root / "Doxyfile", root, env)

    def _device_env(self, env: dict[str, str], device: str | None) -> dict[str, str]:
        """Returns *env* with the selector of *device*, and says so when it replaced one."""
        chosen = with_device(env, device)
        if chosen.replaced is not None:
            self._console.info(
                f"--device overrides {K_DEVICE_VARIABLE}={chosen.replaced} from the environment "
                f"with {chosen.variables[K_DEVICE_VARIABLE]}.",
            )
        return chosen.variables

    def _remove_tree(self, tree: Path) -> bool:
        """Removes *tree* and returns whether it existed.

        Raises:
            TreeNotRemovedError: The tree is still there after the removal.
        """
        if not tree.is_dir():
            return False
        self._console.info(f"Removing {tree}...")
        shutil.rmtree(tree, ignore_errors=True)
        if tree.exists():
            raise TreeNotRemovedError(tree)
        return True

    def _pick_executable(self, build_dir: Path, target: str | None, sort: bool,
                         default: str) -> Path | None:
        """Returns the executable to run; None when the interactive pick found none.

        Raises:
            ExecutableNotFoundError: *target*, or *default*, matches nothing.
        """
        if sort:
            return self._index.select(build_dir, self._console)
        if target:
            exe = self._index.match(build_dir, target)
            if exe is None:
                raise ExecutableNotFoundError(f"Executable matching '{target}'")
            return exe
        exe = self._index.match(build_dir, default)
        if exe is None:
            raise ExecutableNotFoundError(f"Default target '{default}'")
        return exe

    def _configure_args(self, cfg: Config, root: Path, build_dir: Path,
                        build_type: str) -> list[str]:
        """Returns the cmake configure command line for this tree, without caller -D flags."""
        args = [
            self._driver.cmake(cfg), "-S", str(root), "-B", str(build_dir), "-G", cfg.generator,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DSUSHIRUNTIME_DIR={cfg.runtime_dir(root)}",
        ]
        args += _toolchain_args(cfg, root)
        if cfg.is_windows and (prefix := vcpkg_prefix(cfg, root)):
            args.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        return args

    def _test_package(self, cfg: Config, root: Path, build_dir: Path,
                      env: dict[str, str]) -> int:
        """Installs the package and builds and runs tests/package/ against the install prefix.

        Raises:
            StepFailedError: Install, configure, compile or the consumer run failed.
            ExecutableNotFoundError: The consumer built but its executable is absent.
        """
        build_type = cached_value(build_dir, "CMAKE_BUILD_TYPE") or "Release"
        prefix = build_dir / "package" / "prefix"
        consumer = build_dir / "package" / "consumer"

        # The install starts from an empty prefix, so a deleted header cannot satisfy an include.
        self._remove_tree(prefix)

        self._console.info(f"Installing to {prefix}...")
        rc = self._runner.run(
            [self._driver.cmake(cfg), "--install", str(build_dir),
             "--prefix", str(prefix), "--config", build_type],
            root,
            env,
        )
        if rc != 0:
            raise StepFailedError("Install failed; the package could not be staged.", rc)

        prefix_path = str(prefix)
        if extra_prefix := vcpkg_prefix(cfg, root):
            prefix_path = f"{prefix_path};{extra_prefix}"

        self._console.info("Configuring the out-of-tree consumer...")
        configure = [
            self._driver.cmake(cfg), "-S", str(root / "tests" / "package"), "-B", str(consumer),
            "-G", cfg.generator,
            "--no-warn-unused-cli",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_PREFIX_PATH={prefix_path}",
        ]
        configure += _toolchain_args(cfg, root)
        rc = self._driver.configure(configure, cwd=root, env=env)
        if rc != 0:
            raise StepFailedError(
                "The consumer could not configure against the installed package.\n"
                "  - a FATAL_ERROR above about leaked flags means the exported "
                "interface carries development-only options; see "
                "cmake/SushiBLASBuildPolicy.cmake.",
                rc,
            )

        self._console.info("Building the consumer...")
        rc = self._driver.compile(cfg, consumer, cwd=root, env=env, config=build_type)
        if rc != 0:
            raise StepFailedError(
                "The consumer failed to compile against the installed headers.\n"
                "  - a missing header means it is not covered by install(DIRECTORY) "
                "in src/CMakeLists.txt.\n"
                "  - the consumer compiles at C++20 while the library builds at "
                "C++17, so a construct valid only in one of the two fails here.",
                rc,
            )

        exe = self._index.match(consumer, K_PACKAGE_CONSUMER)
        if exe is None:
            raise ExecutableNotFoundError(f"The built consumer executable '{K_PACKAGE_CONSUMER}'")

        if cfg.is_windows:
            count = _deploy_consumer_dlls(cfg, root, prefix, exe.parent)
            self._console.info(f"Deployed {count} shared libraries beside the consumer.")

        self._console.info(f"Executing: {exe.name}")
        rc = self._runner.run([str(exe)], consumer, env)
        if rc != 0:
            raise StepFailedError("The package consumer failed at runtime.", rc)
        return rc


def _build_dir(root: Path) -> Path:
    """Returns the build tree of the checkout at *root*."""
    return root / "build"


def _existing_build_dir(root: Path) -> Path:
    """Returns the build tree of the checkout at *root*.

    Raises:
        BuildTreeMissingError: The tree does not exist.
    """
    build_dir = _build_dir(root)
    if not build_dir.is_dir():
        raise BuildTreeMissingError()
    return build_dir


def _check_device(device: str | None) -> None:
    """Checks that *device* is absent or a value --device accepts.

    Raises:
        InvalidDeviceError: It is neither.
    """
    if device is not None:
        device_selector(device)


def _check_runtime(cfg: Config, root: Path) -> None:
    """Checks that the SushiRuntime sibling holds a CMakeLists.txt.

    Raises:
        RuntimeMissingError: It does not.
    """
    runtime = cfg.runtime_dir(root)
    if not (runtime / "CMakeLists.txt").is_file():
        raise RuntimeMissingError(runtime)


def _toolchain_args(cfg: Config, root: Path) -> list[str]:
    """Returns the compiler and host-tool -D flags both the project and package configures share.

    CMAKE_PREFIX_PATH is not among them; each configure sets its own. See
    cli/sushiblas/README.md.
    """
    cxx = cfg.resolved_compiler(root)
    args = [
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_C_COMPILER={c_compiler_for(cxx)}",
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


def _deploy_consumer_dlls(cfg: Config, root: Path, prefix: Path, dest: Path) -> int:
    """Copies the Windows DLLs the consumer loads beside it and returns their count."""
    sources = list((prefix / "bin").glob("*.dll"))
    if extra_prefix := vcpkg_prefix(cfg, root):
        sources += list(Path(extra_prefix, "bin").glob("hwloc*.dll"))
    for dll in sources:
        shutil.copy2(dll, dest / dll.name)
    return len(sources)
