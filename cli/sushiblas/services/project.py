# project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Runs SushiBLAS's build, test, run, clean and docs through sushicore's CMake driver."""

from __future__ import annotations

import shutil
import sys
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sushicore.build_admission import (
    compiler_cache,
    hardware_threads,
    launcher_defines,
    refuse_a_stopped_gate,
    state_dir,
)
from sushicore.build_env import StackBuildEnv
from sushicore.build_report import ninja_log_offset, of_build
from sushicore.build_tree import refuse_another_build_type
from sushicore.cmake_cache import cached_value
from sushicore.cmake_driver import CMakeDriver
from sushicore.discovery import ExecutableIndex
from sushicore.proc import Runner
from sushicore.provision.sushi_packages import PackageSet, configure_arguments
from sushicore.toolchain_args import c_compiler_for
from sushicore.toolchain_args import vcpkg_prefix

from ..config import Config
from .build_type import K_BENCHMARK_BUILD_TYPE, BuildType
from .errors import (
    BuildTreeMissingError,
    ExecutableNotFoundError,
    InvalidDefineError,
    LeanBenchmarksError,
    StepFailedError,
    TreeNotRemovedError,
)
from . import library_packages
from .sanitizer import Sanitizer, sanitize_flags
from .test_label import TestLabel, select_labels
from .tree_choice import TreeChoice

K_PACKAGE_CONSUMER = "sushiblas_package_consumer"
K_BENCHMARKS_VARIABLE = "SB_BUILD_BENCHMARKS"
K_DISTRIBUTED_VARIABLE = "SB_ENABLE_DISTRIBUTED"
#: What the tree builds beyond the library; `--lean` turns the tests and the examples off.
K_TESTS_VARIABLE = "SB_BUILD_TESTS"
K_EXAMPLES_VARIABLE = "SB_BUILD_EXAMPLES"
#: Where the generated documentation goes, below the root; it belongs to no build tree.
K_DOCS_DIR = Path("build") / "docs"
K_DOXYFILE = Path(".config") / "doxygen" / "Doxyfile"
K_CLI_DIR = "cli"
#: The tree every command but `sb bench` finds when it is not told another.
K_DEFAULT_TREE = TreeChoice()
#: The tree `sb bench` finds when it is not told another: the Release build.
K_BENCHMARK_TREE = TreeChoice(K_BENCHMARK_BUILD_TYPE)
#: Where cmake/SushiBLASTests.cmake places every test and benchmark program, under the build tree.
K_PROGRAM_DIR = "tests"
#: The prefix every benchmark program's name starts with.
K_BENCHMARK_PREFIX = "benchmark_"


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


def distributed_flags(distributed: bool | None) -> list[str]:
    """Returns the -D flag that switches the cluster module and the runtime's distributed layer.

    Args:
        distributed: True builds them, False stops building them, None leaves the cached value.
    """
    if distributed is None:
        return []
    return [f"-D{K_DISTRIBUTED_VARIABLE}={'ON' if distributed else 'OFF'}"]


def lean_flags(lean: bool) -> list[str]:
    """Returns the -D flags of what the tree builds: the tests and the examples, off when lean.

    A lean build also leaves out the benchmarks; :meth:`ProjectService.build` refuses
    ``--benchmarks`` with it, so a plain build keeps the benchmark switch as it is.

    Args:
        lean: True builds no tests, examples or benchmarks.
    """
    state = "OFF" if lean else "ON"
    flags = [f"-D{K_TESTS_VARIABLE}={state}", f"-D{K_EXAMPLES_VARIABLE}={state}"]
    return [*flags, f"-D{K_BENCHMARKS_VARIABLE}=OFF"] if lean else flags


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
        sanitizer: Sanitizer | None = None,
        jobs: int | None = None,
        distributed: bool | None = None,
        lean: bool = False,
        tree_path: str | None = None,
        report: bool = False,
    ) -> int:
        """Configures and builds the tree of *build_type* and *sanitizer*.

        Each build type and sanitizer set has a tree of its own, build/<name>-<type>, so a build
        never empties a tree to change its type. *benchmarks* switches the benchmark programs on
        or off and *distributed* the cluster module; None keeps the cached value of each.
        *jobs* caps the compile jobs run at once; None runs one per hardware thread. *lean*
        builds no tests, examples or benchmarks, which is what a library package is built as, and
        *tree_path* names the tree below the root in place of build/<name>-<type>. *report*
        prints what each kind of build edge cost and writes build_report.json into the tree.

        SushiRuntime and SushiLog are linked as built packages: a missing one is built first, and
        the folders their libraries load from go first on the search path of the build and of
        every later run of the tree.

        Raises:
            InvalidDefineError: An entry of *defines* is not VAR=VALUE; nothing was touched.
            LeanBenchmarksError: *lean* and ``benchmarks=True`` ask for opposite things.
            ConfigError: The tree is configured as another build type, or *defines* sets one.
            BuildStoppedError: The memory gate of the checkout is stopped.
            PackageError: A package of the closure cannot be chosen or built.
            TreeNotRemovedError: A tree that had to go is still there.
            StepFailedError: Configure or compile failed.
        """
        if lean and benchmarks:
            raise LeanBenchmarksError()
        tree = TreeChoice(build_type, sanitizer)
        define_flags = (benchmarks_flags(benchmarks) + distributed_flags(distributed)
                        + sanitize_flags(sanitizer) + validate_defines(defines))
        root = self._find_root()
        cfg = self._load_config()
        build_dir = tree.directory(root, tree_path)

        refuse_another_build_type(build_dir, tree.cmake_type, defines or ())
        refuse_a_stopped_gate(build_dir)
        if clean:
            self._remove_tree(build_dir)

        compiler = cfg.resolved_compiler(root)
        self._console.info(f"Compiler: {compiler}")
        env = self._build_env.load(cfg, build_dir)

        fresh = self._driver.needs_configure(build_dir, cfg.generator, root=root)
        if fresh:
            if build_dir.is_dir():
                self._console.info(
                    f"{build_dir} was configured for another generator or source path; "
                    "starting from an empty tree.",
                )
                self._remove_tree(build_dir)
            self._console.info(f"Configuring CMake... (type={build_type.value})")
        else:
            self._console.info("Reconfiguring in place...")

        packages = library_packages.resolve(
            cfg, root, build_type.value, compiler, _distributed(build_dir, distributed),
            self._console.info, self._console.warn)
        run_dirs = packages.run_dirs(cfg.platform)
        library_packages.prepend_run_dirs(cfg, env, run_dirs)

        configure_args = self._configure_args(cfg, root, build_dir, tree.cmake_type, packages)
        configure_args += lean_flags(lean)
        rc = self._driver.configure(configure_args + define_flags, cwd=root, env=env)
        if rc != 0:
            raise StepFailedError("CMake configure failed.", rc)
        library_packages.record_run_dirs(build_dir, run_dirs)

        self._console.info("Building...")
        offset = ninja_log_offset(build_dir)
        rc = self._driver.compile(cfg, build_dir, cwd=root, env=env, config=tree.cmake_type,
                                  jobs=jobs or hardware_threads())
        if report:
            self._print_report(build_dir, offset)
        if rc != 0:
            raise StepFailedError("Build failed.", rc)
        return rc

    def test(self, labels: Sequence[TestLabel] | None = None, filter: str | None = None,
             repeat: int = 0, jobs: int | None = None, tree: TreeChoice = K_DEFAULT_TREE) -> int:
        """Runs the CTest tests *labels* select and returns ctest's code.

        None or empty *labels* selects the small and medium tests. A quarantined test runs only
        when *labels* names `quarantined`. *tree* picks the build tree that is tested.

        Raises:
            BuildTreeMissingError: There is no build tree.
        """
        root = self._find_root()
        cfg = self._load_config()
        build_dir = _existing_build_dir(root, tree)
        env = self._run_env(cfg, build_dir)
        selection = select_labels(labels)

        return self._driver.ctest_run(
            cfg,
            build_dir,
            env,
            label_regex=selection.ctest_include(),
            exclude_label_regex=selection.ctest_exclude(),
            filter=filter,
            repeat=repeat,
            jobs=jobs,
        )

    def test_cli(self, labels: Sequence[TestLabel] | None = None) -> int:
        """Runs the pytest tests of this CLI that *labels* select and returns pytest's code."""
        root = self._find_root()
        selection = select_labels(labels)
        cmd = [sys.executable, "-m", "pytest", "-q", "-m", selection.pytest_expression()]
        return self._runner.run(cmd, root / K_CLI_DIR, None)

    def test_package(self, tree: TreeChoice = K_DEFAULT_TREE) -> int:
        """Installs the package, builds tests/package/ against it and runs it.

        Raises:
            BuildTreeMissingError: There is no build tree.
            StepFailedError: Install, configure, compile or the consumer run failed.
            ExecutableNotFoundError: The consumer built but its executable is absent.
        """
        root = self._find_root()
        cfg = self._load_config()
        build_dir = _existing_build_dir(root, tree)
        env = self._run_env(cfg, build_dir)
        return self._test_package(cfg, root, build_dir, env)

    def bench(self, name: str | None = None, app_args: Sequence[str] = (),
              tree: TreeChoice = K_BENCHMARK_TREE) -> int:
        """Runs one scenario benchmark with *app_args*, or every micro benchmark program in name order.

        *tree* picks the build tree; a benchmark measures the Release build unless told another.

        Returns the exit code of the program, or the first non-zero code of the programs.

        Raises:
            BuildTreeMissingError: There is no build tree.
            ExecutableNotFoundError: *name* matches no executable, or no benchmark program is built.
        """
        root = self._find_root()
        cfg = self._load_config()
        build_dir = _existing_build_dir(root, tree)
        env = self._run_env(cfg, build_dir)
        if name is not None:
            exe = self._index.match(build_dir, name)
            if exe is None:
                raise ExecutableNotFoundError(f"Benchmark matching '{name}' (build with --benchmarks)")
            self._console.info(f"Executing: {exe.name}")
            return self._runner.run([str(exe), *app_args], root, env)
        programs = _benchmark_programs(build_dir / K_PROGRAM_DIR)
        if not programs:
            raise ExecutableNotFoundError("A benchmark program (build with --benchmarks)")
        code = 0
        for program in programs:
            self._console.info(f"Executing: {program.name}")
            rc = self._runner.run([str(program)], root, env)
            code = code or rc
        return code

    def run(self, target: str | None = None, sort: bool = False,
            app_args: Sequence[str] = (), tree: TreeChoice = K_DEFAULT_TREE) -> int:
        """Runs a built executable with *app_args* and returns its exit code.

        Raises:
            BuildTreeMissingError: There is no build tree.
            ExecutableNotFoundError: *target*, or the configured default, matches nothing.
        """
        root = self._find_root()
        cfg = self._load_config()
        build_dir = _existing_build_dir(root, tree)
        env = self._run_env(cfg, build_dir)

        exe = self._pick_executable(build_dir, target, sort, cfg.target_bin)
        if exe is None:
            return 1

        self._console.info(f"Executing: {exe.name}")
        self._console.console.print(f"[muted]{exe}[/muted]")
        return self._runner.run([str(exe), *app_args], root, env)

    def clean(self, tree: TreeChoice = K_DEFAULT_TREE) -> CleanResult:
        """Removes the build tree *tree* names.

        Raises:
            TreeNotRemovedError: The tree is still there after the removal.
        """
        build_dir = tree.directory(self._find_root())
        return CleanResult(build_dir, self._remove_tree(build_dir))

    def docs(self) -> int:
        """Generates the Doxygen API reference from the repository's Doxyfile."""
        root = self._find_root()
        cfg = self._load_config()
        env = self._build_env.load(cfg, K_DEFAULT_TREE.directory(root))
        (root / K_DOCS_DIR / "api-site").mkdir(parents=True, exist_ok=True)
        return self._driver.doxygen(cfg, root / K_DOXYFILE, root, env)

    def _run_env(self, cfg: Config, build_dir: Path) -> dict[str, str]:
        """Returns the environment a binary of *build_dir* runs under.

        It is the build environment with the run-time folders of the packages the tree linked
        first on the search path.
        """
        env = self._build_env.load(cfg, build_dir)
        library_packages.prepend_run_dirs(cfg, env, library_packages.read_run_dirs(build_dir))
        return env

    def _print_report(self, build_dir: Path, offset: int) -> None:
        """Prints what the build that began at log *offset* cost and writes it into *build_dir*."""
        report = of_build(build_dir, offset, state_dir(build_dir))
        for line in report.lines():
            self._console.info(line)
        self._console.info(f"Report: {report.write(build_dir)}")

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

    def _configure_args(self, cfg: Config, root: Path, build_dir: Path, build_type: str,
                        packages: PackageSet) -> list[str]:
        """Returns the cmake configure command line for this tree, without caller -D flags."""
        args = [
            self._driver.cmake(cfg), "-S", str(root), "-B", str(build_dir), "-G", cfg.generator,
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
        # Each library package reaches CMake as <Package>_DIR; CMake never downloads.
        args += configure_arguments(cfg, root, packages)
        args += _toolchain_args(cfg, root)
        if cfg.is_windows and (prefix := vcpkg_prefix(cfg, root)):
            args.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        # Every compile and link of the tree is admitted by the memory gate.
        args += [f"-D{entry}"
                 for entry in launcher_defines(build_dir, compiler_cache(cfg.ccache_exe))]
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
                "in modules/library/sushiblas/CMakeLists.txt.\n"
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


def _distributed(build_dir: Path, requested: bool | None) -> bool:
    """Returns whether the cluster module is built: *requested*, else what the tree has cached."""
    if requested is not None:
        return requested
    return cached_value(build_dir, K_DISTRIBUTED_VARIABLE) == "ON"


def _existing_build_dir(root: Path, tree: TreeChoice) -> Path:
    """Returns the build tree *tree* names under the checkout at *root*.

    Raises:
        BuildTreeMissingError: The tree does not exist.
    """
    build_dir = tree.directory(root)
    if not build_dir.is_dir():
        raise BuildTreeMissingError(build_dir, tree.build_command())
    return build_dir


def _benchmark_programs(program_dir: Path) -> list[Path]:
    """Returns the benchmark programs under *program_dir*, sorted by name."""
    if not program_dir.is_dir():
        return []
    return sorted(
        path for path in program_dir.iterdir()
        if path.is_file() and path.name.startswith(K_BENCHMARK_PREFIX)
        and path.suffix in ("", ".exe")
    )


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
