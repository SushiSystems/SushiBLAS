# test_project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the project service and the commands that print its results."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import sushicore.proc
from sushicore.proc import Runner
from typer.testing import CliRunner

from sushiblas import cli
from sushiblas.services import project
from sushiblas.services.build_type import BuildType
from sushiblas.services.errors import (
    K_EXIT_USAGE,
    BuildTreeMissingError,
    ExecutableNotFoundError,
    InvalidDefineError,
    StepFailedError,
    TreeNotRemovedError,
)
from sushiblas.services.project import CleanResult, ProjectService
from sushiblas.services.sanitizer import Sanitizer
from sushiblas.services.test_label import TestLabel

pytestmark = pytest.mark.medium

K_CONFIGURE_FAILED = 9
K_CHILD_EXIT = 7
K_INTERRUPTED = 130
K_EXECUTABLE = "quickstart_demo"


class _Console:
    """Records every line the service prints, by kind."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.console = self

    def __getattr__(self, kind: str):
        """Returns a printer that records its text under *kind*."""
        return lambda text="": self.lines.append((kind, str(text)))

    def texts(self, kind: str) -> list[str]:
        """Returns the lines recorded under *kind*."""
        return [text for recorded, text in self.lines if recorded == kind]


class _Driver:
    """Stands in for the driver: asks for a configure, fails it, and lets Doxygen succeed."""

    def __init__(self) -> None:
        self.configured: list[list[str]] = []
        self.documented: list[tuple[Path, Path]] = []
        self.tested: list[dict] = []

    def cmake(self, cfg) -> str:
        """Returns the executable name the configure line starts with."""
        return "cmake"

    def needs_configure(self, build_dir: Path, generator: str, **_) -> bool:
        """Reports that the tree cannot be reused."""
        return True

    def configure(self, args, cwd: Path, env=None) -> int:
        """Records the command line and fails, so the build stops before it compiles."""
        self.configured.append(list(args))
        return K_CONFIGURE_FAILED

    def ctest_run(self, cfg, build_dir: Path, env, **selection) -> int:
        """Records the selection ctest was asked for and succeeds."""
        self.tested.append(selection)
        return 0

    def doxygen(self, cfg, doxyfile: Path, cwd: Path, env=None) -> int:
        """Records the Doxyfile and the working directory and succeeds."""
        self.documented.append((doxyfile, cwd))
        return 0


class _Index:
    """Finds one executable, named K_EXECUTABLE, directly under the build tree."""

    def match(self, build_root: Path, query: str) -> Path | None:
        """Returns the executable when *query* names it."""
        exe = build_root / K_EXECUTABLE
        return exe if query == K_EXECUTABLE and exe.is_file() else None


class _BuildEnv:
    """Returns an empty environment without running a shell."""

    def load(self, cfg, build_dir: Path) -> dict[str, str]:
        """Returns no variables."""
        return {}


class _Packages:
    """Stands in for a package set that holds no library."""

    def names(self) -> list[str]:
        """Returns no name."""
        return []

    def definitions(self) -> list[str]:
        """Returns no definition."""
        return []

    def run_dirs(self, platform: str) -> list[str]:
        """Returns no folder."""
        return []


class _Config:
    """Holds the few settings the service reads."""

    generator = "Ninja"
    is_windows = False
    platform = "linux"
    target_bin = K_EXECUTABLE

    def sushi_definitions(self, root: Path, packaged=None) -> list[str]:
        """Returns no definition: every pinned library is a package."""
        return []

    def sushi_packages(self, root: Path, build_type: str, options=None, report=print,
                       warn=print, compiler: str = ""):
        """Returns a package set with no library."""
        return _Packages()

    def resolved_compiler(self, root: Path) -> str:
        """Returns a compiler name; nothing runs it."""
        return "clang++"

    def resolved_vcpkg(self, root: Path) -> str:
        """Returns no vcpkg tree."""
        return ""


@pytest.fixture
def checkout(tmp_path):
    """Returns a checkout with a build tree and a SushiRuntime sibling."""
    root = tmp_path / "sushiblas"
    (root / "build").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("")
    (tmp_path / "sushiruntime").mkdir()
    (tmp_path / "sushiruntime" / "CMakeLists.txt").write_text("")
    return root


@pytest.fixture
def parts(checkout):
    """Returns the recording console and driver and the service built on them."""
    console = _Console()
    driver = _Driver()
    service = ProjectService(
        console=console,
        runner=Runner(console, "sb", catch_interrupt=True),
        driver=driver,
        index=_Index(),
        build_env=_BuildEnv(),
        find_root=lambda: checkout,
        load_config=_Config,
    )
    return console, driver, service


def _child_exits_with(monkeypatch, code: int) -> list[list[str]]:
    """Makes every child process exit with *code* and returns the list of commands run."""
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        """Records *cmd* and returns its fixed exit code."""
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(sushicore.proc.subprocess, "run", fake_run)
    return commands


def test_build_says_it_replaces_a_tree_it_cannot_reuse(parts, checkout):
    """A tree configured for another build type is removed out loud, not silently."""
    console, driver, service = parts
    with pytest.raises(StepFailedError) as raised:
        service.build(BuildType.DEBUG)
    assert raised.value.exit_code == K_CONFIGURE_FAILED
    assert not (checkout / "build").exists()
    said = console.texts("info")
    reason = next(text for text in said if "empty tree" in text)
    assert said.index(reason) < said.index(f"Removing {checkout / 'build'}...")


def test_build_removes_nothing_when_there_is_no_tree(parts, checkout):
    """A first build has no tree to announce or remove."""
    console, driver, service = parts
    (checkout / "build").rmdir()
    with pytest.raises(StepFailedError):
        service.build(BuildType.DEBUG)
    assert not [text for text in console.texts("info") if "empty tree" in text]
    assert not [text for text in console.texts("info") if text.startswith("Removing")]


def test_build_rejects_a_bad_define_before_touching_the_tree(parts, checkout):
    """A -D without '=' raises with exit 2 and leaves the tree and its files in place."""
    console, driver, service = parts
    marker = checkout / "build" / "CMakeCache.txt"
    marker.write_text("")
    with pytest.raises(InvalidDefineError) as raised:
        service.build(BuildType.DEBUG, clean=True, defines=["SB_ENABLE_LOGGING=OFF", "NOVALUE"])
    assert raised.value.exit_code == K_EXIT_USAGE
    assert marker.is_file()
    assert driver.configured == []
    assert console.lines == []


def test_build_appends_defines_after_every_default(parts, checkout):
    """Caller -D flags end the configure line, so no default outranks them."""
    console, driver, service = parts
    with pytest.raises(StepFailedError):
        service.build(BuildType.RELEASE, defines=["SB_ENABLE_LOGGING=OFF", "SB_NATIVE_ARCH="])
    assert driver.configured[0][-2:] == ["-DSB_ENABLE_LOGGING=OFF", "-DSB_NATIVE_ARCH="]


@pytest.mark.parametrize(
    ("benchmarks", "expected"),
    [(True, ["-DSB_BUILD_BENCHMARKS=ON"]), (False, ["-DSB_BUILD_BENCHMARKS=OFF"]), (None, [])],
)
def test_build_passes_the_benchmarks_switch_as_a_cache_variable(
    parts, checkout, benchmarks, expected
):
    """--benchmarks and --no-benchmarks become one SB_BUILD_BENCHMARKS flag; none leaves it out."""
    console, driver, service = parts
    with pytest.raises(StepFailedError):
        service.build(BuildType.RELEASE, benchmarks=benchmarks)
    flags = [arg for arg in driver.configured[0] if arg.startswith("-DSB_BUILD_BENCHMARKS")]
    assert flags == expected


def test_build_lets_a_caller_define_outrank_the_benchmarks_switch(parts, checkout):
    """A -D for the same variable comes after --benchmarks, so it wins."""
    console, driver, service = parts
    with pytest.raises(StepFailedError):
        service.build(
            BuildType.RELEASE, defines=["SB_BUILD_BENCHMARKS=OFF"], benchmarks=True
        )
    configured = driver.configured[0]
    assert configured.index("-DSB_BUILD_BENCHMARKS=ON") < configured.index(
        "-DSB_BUILD_BENCHMARKS=OFF"
    )


@pytest.mark.parametrize(
    ("sanitizer", "expected"),
    [
        (Sanitizer.ADDRESS_UNDEFINED, ["-DSB_SANITIZE=address,undefined"]),
        (Sanitizer.THREAD, ["-DSB_SANITIZE=thread"]),
        (Sanitizer.NONE, ["-DSB_SANITIZE="]),
        (None, []),
    ],
)
def test_build_passes_the_sanitizer_set_as_a_cache_variable(parts, checkout, sanitizer, expected):
    """--sanitize becomes one SB_SANITIZE flag; none clears it and leaving it out keeps it."""
    console, driver, service = parts
    with pytest.raises(StepFailedError):
        service.build(BuildType.RELEASE, sanitizer=sanitizer)
    flags = [arg for arg in driver.configured[0] if arg.startswith("-DSB_SANITIZE")]
    assert flags == expected


class _CompilingDriver(_Driver):
    """Stands in for the driver: configures, then records the jobs each compile was given."""

    def __init__(self) -> None:
        super().__init__()
        self.compiled_jobs: list[int | None] = []

    def configure(self, args, cwd: Path, env=None) -> int:
        """Records the command line and succeeds, so the build goes on to compile."""
        self.configured.append(list(args))
        return 0

    def compile(self, cfg, build_dir: Path, cwd: Path, env, **options) -> int:
        """Records the jobs the compile was asked for and succeeds."""
        self.compiled_jobs.append(options.get("jobs"))
        return 0


def test_build_hands_the_jobs_to_the_compile(checkout):
    """The number of compile jobs reaches the driver as given, and is absent when not given."""
    console = _Console()
    driver = _CompilingDriver()
    service = ProjectService(
        console=console,
        runner=Runner(console, "sb", catch_interrupt=True),
        driver=driver,
        index=_Index(),
        build_env=_BuildEnv(),
        find_root=lambda: checkout,
        load_config=_Config,
    )
    assert service.build(BuildType.RELEASE, jobs=4) == 0
    assert service.build(BuildType.RELEASE) == 0
    assert driver.compiled_jobs == [4, None]


def test_test_runs_small_and_medium_and_leaves_quarantined_out_by_default(parts, checkout):
    """With no label, ctest selects the small and medium labels and excludes quarantined."""
    console, driver, service = parts
    assert service.test() == 0
    assert driver.tested[0]["label_regex"] == "^(small|medium)$"
    assert driver.tested[0]["exclude_label_regex"] == "^quarantined$"


def test_test_passes_no_exclusion_when_quarantined_is_named(parts, checkout):
    """`--label quarantined` selects the quarantined tests and excludes nothing."""
    console, driver, service = parts
    service.test([TestLabel.QUARANTINED])
    assert driver.tested[0]["label_regex"] == "^quarantined$"
    assert driver.tested[0]["exclude_label_regex"] is None


def test_test_hands_the_jobs_to_ctest(parts, checkout):
    """The number of tests run at once reaches ctest as given, and is absent when not given."""
    console, driver, service = parts
    service.test(jobs=3)
    service.test()
    assert [run["jobs"] for run in driver.tested] == [3, None]


def test_test_cli_runs_pytest_with_the_marker_expression(parts, checkout, monkeypatch):
    """--cli runs pytest from cli/ with the expression of the labels given."""
    console, driver, service = parts
    commands = _child_exits_with(monkeypatch, 0)
    assert service.test_cli([TestLabel.SMALL]) == 0
    assert commands[0][1:] == ["-m", "pytest", "-q", "-m", "(small) and not quarantined"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["build", "--benchmarks"], True),
        (["build", "--no-benchmarks"], False),
        (["build"], None),
    ],
)
def test_build_command_forwards_the_benchmarks_option(monkeypatch, args, expected):
    """The build command hands the tri-state option to the service unchanged."""
    received = []

    class _Service:
        def build(self, build_type, clean, define, benchmarks, sanitize, jobs, distributed, lean,
                  tree_path):
            received.append(benchmarks)
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0
    assert received == [expected]


@pytest.mark.parametrize(("args", "expected"), [(["build", "--jobs", "4"], 4), (["build"], None)])
def test_build_command_forwards_the_jobs_option(monkeypatch, args, expected):
    """The build command hands --jobs to the service unchanged, and None when it is left out."""
    received = []

    class _Service:
        def build(self, build_type, clean, define, benchmarks, sanitize, jobs, distributed, lean,
                  tree_path):
            received.append(jobs)
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0
    assert received == [expected]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["build", "--distributed"], True),
        (["build", "--no-distributed"], False),
        (["build"], None),
    ],
)
def test_build_command_forwards_the_distributed_option(monkeypatch, args, expected):
    """The build command hands the tri-state --distributed option to the service unchanged."""
    received = []

    class _Service:
        def build(self, build_type, clean, define, benchmarks, sanitize, jobs, distributed, lean,
                  tree_path):
            received.append(distributed)
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0
    assert received == [expected]


def test_distributed_flags_switch_the_cluster_module():
    """distributed_flags gives the SB_ENABLE_DISTRIBUTED define, or nothing for None."""
    assert project.distributed_flags(True) == ["-DSB_ENABLE_DISTRIBUTED=ON"]
    assert project.distributed_flags(False) == ["-DSB_ENABLE_DISTRIBUTED=OFF"]
    assert project.distributed_flags(None) == []


def test_clean_reports_a_removed_tree(parts, checkout):
    """clean returns the tree and True when it removed one."""
    console, driver, service = parts
    assert service.clean() == CleanResult(checkout / "build", True)
    assert not (checkout / "build").exists()


def test_clean_reports_a_missing_tree(parts, checkout):
    """clean returns False, not an error, when there is no tree."""
    console, driver, service = parts
    (checkout / "build").rmdir()
    assert service.clean() == CleanResult(checkout / "build", False)


def test_clean_raises_when_the_tree_remains(parts, checkout, monkeypatch):
    """A tree the removal could not delete raises instead of reporting success."""
    console, driver, service = parts
    monkeypatch.setattr(project.shutil, "rmtree", lambda path, ignore_errors=False: None)
    with pytest.raises(TreeNotRemovedError) as raised:
        service.clean()
    assert raised.value.exit_code == 1
    assert (checkout / "build").is_dir()


def test_clean_command_exits_one_when_the_tree_remains(parts, checkout, monkeypatch):
    """sb clean prints the failure and exits 1, with no success line."""
    console, driver, service = parts
    monkeypatch.setattr(project.shutil, "rmtree", lambda path, ignore_errors=False: None)
    monkeypatch.setattr(cli, "_service", lambda: service)
    printed = _Console()
    monkeypatch.setattr(cli, "console", printed)
    result = CliRunner().invoke(cli.app, ["clean"])
    assert result.exit_code == 1
    assert len(printed.texts("error")) == 1
    assert str(checkout / "build") in printed.texts("error")[0]
    assert printed.texts("success") == []


def test_run_returns_the_exit_code_of_the_child(parts, checkout, monkeypatch):
    """run launches the default target with the forwarded arguments and returns its code."""
    console, driver, service = parts
    exe = checkout / "build" / K_EXECUTABLE
    exe.write_text("")
    commands = _child_exits_with(monkeypatch, K_CHILD_EXIT)
    assert service.run(app_args=["--flag"]) == K_CHILD_EXIT
    assert commands == [[str(exe), "--flag"]]


def test_run_raises_for_a_target_that_matches_nothing(parts):
    """An unknown target raises instead of returning a bare 1."""
    console, driver, service = parts
    with pytest.raises(ExecutableNotFoundError):
        service.run(target="no_such_binary")


def test_run_raises_without_a_build_tree(parts, checkout):
    """run needs a build tree."""
    console, driver, service = parts
    (checkout / "build").rmdir()
    with pytest.raises(BuildTreeMissingError):
        service.run()


@pytest.mark.parametrize("args", [["run", "--sortt"], ["run", "x", "--bogus"], ["run", "--", "-x"]])
def test_run_command_rejects_a_mistyped_option(monkeypatch, args):
    """An option sb run does not know is a usage error, never a target or a forwarded flag."""

    def no_service():
        """Fails the test if the command reaches the service."""
        raise AssertionError("the service was reached")

    monkeypatch.setattr(cli, "_service", no_service)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == K_EXIT_USAGE


@pytest.mark.parametrize("args", [
    ["test", "--package", "--filter", "Gemm.*"],
    ["test", "--package", "--label", "small"],
    ["test", "--package", "--cli"],
    ["test", "--cli", "--repeat", "2"],
    ["test", "--cli", "--jobs", "2"],
    ["test", "--package", "--jobs", "2"],
])
def test_test_command_refuses_options_that_do_not_apply(monkeypatch, args):
    """--package takes no selection option and --cli no ctest option; both exit 2."""

    def no_service():
        """Fails the test if the command reaches the service."""
        raise AssertionError("the service was reached")

    monkeypatch.setattr(cli, "_service", no_service)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == K_EXIT_USAGE


def test_docs_creates_the_site_folder_under_the_build_tree(parts, checkout):
    """docs makes build/docs/api-site and hands the driver the Doxyfile under .config."""
    console, driver, service = parts
    assert service.docs() == 0
    assert (checkout / "build" / "docs" / "api-site").is_dir()
    assert not (checkout / "docs").exists()
    assert driver.documented == [(checkout / ".config" / "doxygen" / "Doxyfile", checkout)]


def test_an_interrupted_child_ends_in_exit_code_130(tmp_path, monkeypatch):
    """Ctrl+C during a child process becomes a warning and 130, not a traceback."""

    def interrupt(*args, **kwargs):
        """Raises what Ctrl+C raises while the child runs."""
        raise KeyboardInterrupt

    console = _Console()
    runner = Runner(console, "sb", catch_interrupt=True)
    monkeypatch.setattr(sushicore.proc.subprocess, "run", interrupt)
    assert runner.run(["some-tool"], tmp_path, {}) == K_INTERRUPTED
    assert console.texts("warn") == ["Interrupted."]


def test_bench_raises_when_no_benchmark_program_is_built(parts, checkout):
    """bench with no benchmark program says how to build one."""
    console, driver, service = parts
    with pytest.raises(ExecutableNotFoundError):
        service.bench()


def test_bench_runs_every_benchmark_program_in_name_order(parts, checkout, monkeypatch):
    """bench runs each benchmark_* program under build/tests and skips the test programs."""
    console, driver, service = parts
    programs = checkout / "build" / "tests"
    programs.mkdir()
    for name in ("benchmark_sparse", "benchmark_gemm", "sushiblas_blas_tests"):
        (programs / name).write_text("")
    commands = _child_exits_with(monkeypatch, 0)
    assert service.bench() == 0
    assert [Path(command[0]).name for command in commands] == ["benchmark_gemm", "benchmark_sparse"]


def test_bench_runs_the_named_scenario_benchmark_with_the_forwarded_options(
    parts, checkout, monkeypatch
):
    """bench with a name runs that one executable and passes the options after it."""
    console, driver, service = parts
    exe = checkout / "build" / K_EXECUTABLE
    exe.write_text("")
    commands = _child_exits_with(monkeypatch, K_CHILD_EXIT)
    assert service.bench(K_EXECUTABLE, ["--out", "folder"]) == K_CHILD_EXIT
    assert commands == [[str(exe), "--out", "folder"]]


def test_bench_raises_for_a_scenario_benchmark_that_matches_nothing(parts, checkout):
    """bench with a name that matches no executable raises instead of running every program."""
    console, driver, service = parts
    with pytest.raises(ExecutableNotFoundError):
        service.bench("no_such_benchmark")


def test_bench_command_forwards_the_name_and_the_options_after_the_separator(monkeypatch):
    """The bench command hands the name and what follows `--` to the service."""
    received = []

    class _Service:
        def bench(self, name, app_args):
            received.append((name, app_args))
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, ["bench", "device_ceilings", "--", "--runs", "3"])
    assert result.exit_code == 0
    assert received == [("device_ceilings", ["--runs", "3"])]


def test_bench_command_without_a_name_runs_every_program(monkeypatch):
    """The bench command with no name hands the service no name and no options."""
    received = []

    class _Service:
        def bench(self, name, app_args):
            received.append((name, app_args))
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, ["bench"])
    assert result.exit_code == 0
    assert received == [(None, [])]


@pytest.mark.parametrize("args", [["bench", "--", "--runs", "3"], ["bench", "--sortt"]])
def test_bench_command_rejects_options_that_belong_to_no_benchmark(monkeypatch, args):
    """Options need a benchmark name before the separator; an unknown flag is a usage error."""

    def no_service():
        """Fails the test if the command reaches the service."""
        raise AssertionError("the service was reached")

    monkeypatch.setattr(cli, "_service", no_service)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == K_EXIT_USAGE
