# test_project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the project service and the commands that print its results, over a scratch checkout."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from command_harness import K_DEBUG_TREE, K_EXECUTABLE, K_RELEASE_TREE, Commands
from sushicore.build_admission import BuildStoppedError, hardware_threads, stop_file
from sushicore.console_memory import RecordingConsole
from sushicore.errors import ConfigError
from sushicore.proc import Runner
from sushicore.proc_memory import RecordingRunner
from sushicore.testing import TestLabel

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

pytestmark = pytest.mark.medium

K_CONFIGURE_FAILED = 9
K_CHILD_EXIT = 7
K_INTERRUPTED = 130
K_COMPILER_LAUNCHER = "CMAKE_CXX_COMPILER_LAUNCHER"
K_LINKER_LAUNCHER = "CMAKE_CXX_LINKER_LAUNCHER"
K_GATE_LAUNCHER_END = ";--"


@pytest.fixture
def checkout(commands: Commands) -> Path:
    """Returns the scratch checkout of the commands fixture."""
    return commands.root


@pytest.fixture
def runner(commands: Commands) -> RecordingRunner:
    """Returns the runner that records each command the service runs and starts no process."""
    return commands.runner


@pytest.fixture
def console(commands: Commands) -> RecordingConsole:
    """Returns the console that records every line the service prints."""
    return commands.console


@pytest.fixture
def service(commands: Commands) -> ProjectService:
    """Returns the real service of the commands fixture, on sushicore's CMake driver."""
    return commands.context.service_factory()


def _option(argv: tuple[str, ...], flag: str) -> str | None:
    """Returns the word after *flag* in *argv*, or None when *flag* is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def _definitions(argv: tuple[str, ...]) -> dict[str, str]:
    """Returns every cache entry *argv* sets with ``-DNAME=VALUE``, by name."""
    return dict(word[2:].split("=", 1) for word in argv if word.startswith("-D") and "=" in word)


def _flags_starting(argv: tuple[str, ...], prefix: str) -> list[str]:
    """Returns the words of *argv* that start with *prefix*, in order."""
    return [word for word in argv if word.startswith(prefix)]


def _keep_tree(root: Path) -> None:
    """Leaves the Debug tree of the scratch checkout in place."""


def _remove_debug_tree(root: Path) -> None:
    """Removes the empty Debug tree of the scratch checkout."""
    (root / K_DEBUG_TREE).rmdir()


def test_build_says_it_replaces_a_tree_of_another_generator(service, runner, console, checkout):
    """A tree the driver cannot reuse is removed out loud, not silently."""
    runner.script(K_CONFIGURE_FAILED)
    with pytest.raises(StepFailedError) as raised:
        service.build(BuildType.DEBUG)
    assert raised.value.exit_code == K_CONFIGURE_FAILED
    assert not (checkout / K_DEBUG_TREE).exists()
    said = console.texts("info")
    reason = next(text for text in said if "empty tree" in text)
    assert said.index(reason) < said.index(f"Removing {checkout / K_DEBUG_TREE}...")


def test_build_removes_nothing_when_there_is_no_tree(service, runner, console, checkout):
    """A first build has no tree to announce or remove."""
    _remove_debug_tree(checkout)
    runner.script(K_CONFIGURE_FAILED)
    with pytest.raises(StepFailedError):
        service.build(BuildType.DEBUG)
    assert not [text for text in console.texts("info") if "empty tree" in text]
    assert not [text for text in console.texts("info") if text.startswith("Removing")]


def test_build_rejects_a_bad_define_before_touching_the_tree(service, runner, console, checkout):
    """A -D without '=' raises with exit 2 and leaves the tree and its files in place."""
    marker = checkout / K_DEBUG_TREE / "CMakeCache.txt"
    marker.write_text("")
    with pytest.raises(InvalidDefineError) as raised:
        service.build(BuildType.DEBUG, clean=True, defines=["SB_ENABLE_LOGGING=OFF", "NOVALUE"])
    assert raised.value.exit_code == K_EXIT_USAGE
    assert marker.is_file()
    assert runner.commands == []
    assert console.calls == []


def test_build_appends_defines_after_every_default(service, runner):
    """Caller -D flags end the configure line, so no default outranks them."""
    service.build(BuildType.RELEASE, defines=["SB_ENABLE_LOGGING=OFF", "SB_NATIVE_ARCH="])
    assert runner.commands[0].argv[-2:] == ("-DSB_ENABLE_LOGGING=OFF", "-DSB_NATIVE_ARCH=")


@pytest.mark.parametrize(("jobs", "expected"), [(4, "4"), (None, str(hardware_threads()))])
def test_build_hands_the_jobs_to_the_compile(service, runner, jobs, expected):
    """The jobs given reach the compile as given, and one per hardware thread when not given."""
    assert service.build(BuildType.RELEASE, jobs=jobs) == 0
    assert _option(runner.commands[1].argv, "--parallel") == expected


@pytest.mark.parametrize(
    ("build_type", "tree", "cmake_type"),
    [(BuildType.DEBUG, K_DEBUG_TREE, "Debug"), (BuildType.RELEASE, K_RELEASE_TREE, "Release")],
)
def test_build_configures_the_tree_of_its_type(service, runner, checkout, build_type, tree,
                                               cmake_type):
    """The configure line names the tree of the type and its CMake spelling."""
    service.build(build_type)
    configured = runner.commands[0].argv
    assert _option(configured, "-B") == str(checkout / tree)
    assert _definitions(configured)["CMAKE_BUILD_TYPE"] == cmake_type


def test_build_refuses_to_configure_a_tree_of_another_type(service, runner, checkout):
    """A tree configured as Release is not configured as Debug, and nothing is removed."""
    cache = checkout / K_DEBUG_TREE / "CMakeCache.txt"
    cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
    with pytest.raises(ConfigError):
        service.build(BuildType.DEBUG)
    assert cache.is_file()
    assert runner.commands == []


def test_build_refuses_a_define_that_sets_the_build_type(service, runner):
    """-D CMAKE_BUILD_TYPE would split the tree from its name, so it is refused."""
    with pytest.raises(ConfigError):
        service.build(BuildType.DEBUG, defines=["CMAKE_BUILD_TYPE=Release"])
    assert runner.commands == []


def test_build_refuses_while_the_memory_gate_is_stopped(service, runner, checkout):
    """The stop file of the gate ends the build before it configures."""
    stop = stop_file(checkout / K_DEBUG_TREE)
    stop.parent.mkdir(parents=True)
    stop.write_text("")
    with pytest.raises(BuildStoppedError):
        service.build(BuildType.DEBUG)
    assert runner.commands == []


def test_build_configure_carries_both_gate_launchers(service, runner):
    """The configure line runs every compile and every link through the one gate launcher."""
    service.build(BuildType.DEBUG)
    entries = _definitions(runner.commands[0].argv)
    assert entries[K_COMPILER_LAUNCHER] == entries[K_LINKER_LAUNCHER]
    assert entries[K_COMPILER_LAUNCHER].endswith(K_GATE_LAUNCHER_END)


def test_build_writes_no_report_unless_asked(service, console, checkout):
    """A build without --report prints no report line and leaves no report file."""
    service.build(BuildType.DEBUG)
    assert not [text for text in console.texts("info") if text.startswith("Report:")]
    assert not (checkout / K_DEBUG_TREE / "build_report.json").exists()


def test_build_prints_and_writes_the_report_on_request(service, console, checkout):
    """--report prints the report lines and writes build_report.json into the tree."""
    service.build(BuildType.DEBUG, report=True)
    written = checkout / K_DEBUG_TREE / "build_report.json"
    assert written.is_file()
    assert f"Report: {written}" in console.texts("info")


@pytest.mark.parametrize(
    ("labels", "included", "excluded"),
    [
        (None, "^(small|medium)$", "^quarantined$"),
        ([TestLabel.QUARANTINED], "^quarantined$", None),
    ],
)
def test_test_selects_the_labels_and_leaves_quarantined_out_unless_named(
    service, runner, labels, included, excluded
):
    """No label runs small and medium without quarantined; naming quarantined runs it and excludes nothing."""
    assert service.test(labels) == 0
    ctest = runner.commands[0].argv
    assert _option(ctest, "-L") == included
    assert _option(ctest, "-LE") == excluded


@pytest.mark.parametrize(("jobs", "expected"), [(3, "3"), (None, None)])
def test_test_hands_the_jobs_to_ctest(service, runner, jobs, expected):
    """The number of tests run at once reaches ctest as given, and is absent when not given."""
    service.test(jobs=jobs)
    assert _option(runner.commands[0].argv, "--parallel") == expected


def test_test_cli_runs_pytest_with_the_marker_expression(service, runner, checkout):
    """--cli runs pytest from cli/ with the expression of the labels given."""
    assert service.test_cli([TestLabel.SMALL]) == 0
    assert runner.commands[0].argv[1:] == (
        "-m", "pytest", "-q", "-m", "(small) and not quarantined")
    assert runner.commands[0].cwd == checkout / "cli"


def _forwarded(commands: Commands) -> dict[str, list[str] | str | None]:
    """Returns what one build command handed cmake: its two switches and the compile's jobs."""
    configure, compile_line = commands.runner.commands[0].argv, commands.runner.commands[1].argv
    return {
        "benchmarks": _flags_starting(configure, "-DSB_BUILD_BENCHMARKS"),
        "distributed": _flags_starting(configure, "-DSB_ENABLE_DISTRIBUTED"),
        "jobs": _option(compile_line, "--parallel"),
    }


@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (["build", "--benchmarks"], "benchmarks", ["-DSB_BUILD_BENCHMARKS=ON"]),
        (["build", "--no-benchmarks"], "benchmarks", ["-DSB_BUILD_BENCHMARKS=OFF"]),
        (["build"], "benchmarks", []),
        (["build", "--distributed"], "distributed", ["-DSB_ENABLE_DISTRIBUTED=ON"]),
        (["build", "--no-distributed"], "distributed", ["-DSB_ENABLE_DISTRIBUTED=OFF"]),
        (["build"], "distributed", []),
        (["build", "--jobs", "4"], "jobs", "4"),
        (["build"], "jobs", str(hardware_threads())),
    ],
)
def test_build_command_forwards_each_option_to_the_service(commands, args, key, expected):
    """The build command hands --benchmarks, --distributed and --jobs to the service unchanged."""
    result = commands.invoke(*args)
    assert result.exit_code == 0, result.output
    assert _forwarded(commands)[key] == expected


def test_build_command_defaults_to_debug_with_no_report(commands):
    """With no option the build command asks for the Debug tree and prints no report."""
    assert commands.invoke("build").exit_code == 0
    assert _definitions(commands.runner.commands[0].argv)["CMAKE_BUILD_TYPE"] == "Debug"
    assert not [text for text in commands.console.texts("info") if text.startswith("Report:")]


def test_build_command_forwards_the_report_option(commands):
    """--report reaches the service, which writes the report into the tree of the type asked for."""
    assert commands.invoke("build", "--report", "--type", "release").exit_code == 0
    assert f"Report: {commands.root / K_RELEASE_TREE / 'build_report.json'}" in (
        commands.console.texts("info"))


@pytest.mark.parametrize(
    ("prepare", "removed"),
    [(_keep_tree, True), (_remove_debug_tree, False)],
)
def test_clean_reports_whether_it_removed_the_tree(
    service, checkout, prepare: Callable[[Path], None], removed
):
    """clean returns the tree and whether it removed one; a missing tree is not an error."""
    prepare(checkout)
    assert service.clean() == CleanResult(checkout / K_DEBUG_TREE, removed)
    assert not (checkout / K_DEBUG_TREE).exists()
    assert (checkout / K_RELEASE_TREE).is_dir()


def test_clean_raises_when_the_tree_remains(service, checkout, monkeypatch):
    """A tree the removal could not delete raises instead of reporting success."""
    monkeypatch.setattr(shutil, "rmtree", lambda path, ignore_errors=False: None)
    with pytest.raises(TreeNotRemovedError) as raised:
        service.clean()
    assert raised.value.exit_code == 1
    assert (checkout / K_DEBUG_TREE).is_dir()


def test_clean_command_exits_one_when_the_tree_remains(commands, monkeypatch):
    """sb clean prints the failure and exits 1, with no success line."""
    monkeypatch.setattr(shutil, "rmtree", lambda path, ignore_errors=False: None)
    result = commands.invoke("clean")
    assert result.exit_code == 1
    assert len(commands.console.texts("error")) == 1
    assert str(commands.root / K_DEBUG_TREE) in commands.console.texts("error")[0]
    assert commands.console.texts("success") == []


def test_run_returns_the_exit_code_of_the_child(service, runner, checkout):
    """run launches the default target with the forwarded arguments and returns its code."""
    runner.script(K_CHILD_EXIT)
    assert service.run(app_args=["--flag"]) == K_CHILD_EXIT
    assert [command.argv for command in runner.commands] == [
        (str(checkout / K_DEBUG_TREE / K_EXECUTABLE), "--flag")]


def test_run_raises_for_a_target_that_matches_nothing(service):
    """An unknown target raises instead of returning a bare 1."""
    with pytest.raises(ExecutableNotFoundError):
        service.run(target="no_such_binary")


def test_run_raises_without_a_build_tree(service, checkout):
    """run needs the build tree it is asked for, and says which command makes it."""
    _remove_debug_tree(checkout)
    with pytest.raises(BuildTreeMissingError) as raised:
        service.run()
    assert "sb build --type debug" in str(raised.value)
    assert str(checkout / K_DEBUG_TREE) in str(raised.value)


@pytest.mark.parametrize("args", [["run", "--sortt"], ["run", "x", "--bogus"], ["run", "--", "-x"]])
def test_run_command_rejects_a_mistyped_option(commands, args):
    """An option sb run does not know is a usage error, never a target or a forwarded flag."""
    result = commands.invoke(*args)
    assert result.exit_code == K_EXIT_USAGE
    assert commands.runner.commands == []


@pytest.mark.parametrize("args", [
    ["test", "--package", "--filter", "Gemm.*"],
    ["test", "--package", "--label", "small"],
    ["test", "--package", "--cli"],
    ["test", "--cli", "--repeat", "2"],
    ["test", "--cli", "--jobs", "2"],
    ["test", "--package", "--jobs", "2"],
])
def test_test_command_refuses_options_that_do_not_apply(commands, args):
    """--package takes no selection option and --cli no ctest option; both exit 2."""
    result = commands.invoke(*args)
    assert result.exit_code == K_EXIT_USAGE
    assert commands.runner.commands == []


def test_docs_creates_the_site_folder_under_the_build_tree(service, runner, checkout):
    """docs makes build/docs/api-site and runs Doxygen on the Doxyfile under .config from the root."""
    assert service.docs() == 0
    assert (checkout / "build" / "docs" / "api-site").is_dir()
    assert not (checkout / "docs").exists()
    assert runner.commands[0].argv == (str(checkout.parent / "doxygen"), ".config/doxygen/Doxyfile")
    assert runner.commands[0].cwd == checkout


def test_an_interrupted_child_ends_in_exit_code_130(tmp_path, monkeypatch):
    """Ctrl+C during a child process becomes a warning and 130, not a traceback."""

    def interrupt(*args, **kwargs):
        """Raises what Ctrl+C raises while the child runs."""
        raise KeyboardInterrupt

    console = RecordingConsole()
    runner = Runner(console, "sb", catch_interrupt=True)
    monkeypatch.setattr(subprocess, "run", interrupt)
    assert runner.run(["some-tool"], tmp_path, {}) == K_INTERRUPTED
    assert console.texts("warn") == ["Interrupted."]


def test_bench_raises_when_no_benchmark_program_is_built(service):
    """bench with no benchmark program says how to build one."""
    with pytest.raises(ExecutableNotFoundError):
        service.bench()


def test_bench_runs_every_benchmark_program_in_name_order(service, runner, checkout):
    """bench runs each benchmark_* program under build/tests, skipping the test programs and the ones that take arguments."""
    programs = checkout / K_RELEASE_TREE / "tests"
    programs.mkdir()
    (programs / "benchmark_sparse").write_text("")
    (programs / "benchmark_gemm").write_text("")
    (programs / "benchmark_gemm_blocking").write_text("")
    (programs / "sushiblas_operations_tests").write_text("")
    assert service.bench() == 0
    assert [command.argv for command in runner.commands] == [
        (str(programs / "benchmark_gemm"),), (str(programs / "benchmark_sparse"),)]


def test_bench_runs_the_named_scenario_benchmark_with_the_forwarded_options(
    service, runner, checkout
):
    """bench with a name runs that one executable and passes the options after it."""
    runner.script(K_CHILD_EXIT)
    assert service.bench(K_EXECUTABLE, ["--out", "folder"]) == K_CHILD_EXIT
    assert [command.argv for command in runner.commands] == [
        (str(checkout / K_RELEASE_TREE / K_EXECUTABLE), "--out", "folder")]


def test_bench_raises_for_a_scenario_benchmark_that_matches_nothing(service, runner):
    """bench with a name that matches no executable raises instead of running every program."""
    with pytest.raises(ExecutableNotFoundError):
        service.bench("no_such_benchmark")
    assert runner.commands == []


def test_bench_finds_the_release_tree_unless_told_another(service, checkout):
    """A benchmark measures the build that ships: its default tree is the Release one."""
    (checkout / K_RELEASE_TREE).rmdir()
    with pytest.raises(BuildTreeMissingError) as raised:
        service.bench()
    assert "sb build --type release" in str(raised.value)


def _argv_below(commands: Commands) -> list[tuple[str, ...]]:
    """Returns each command run, its program named relative to the Release tree in POSIX form."""
    tree = commands.root / K_RELEASE_TREE
    return [(Path(command.argv[0]).relative_to(tree).as_posix(), *command.argv[1:])
            for command in commands.runner.commands]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["bench", K_EXECUTABLE, "--", "--runs", "3"], [(K_EXECUTABLE, "--runs", "3")]),
        (["bench"], [("tests/benchmark_gemm",)]),
    ],
)
def test_bench_command_forwards_the_name_and_the_options_after_the_separator(
    commands, args, expected
):
    """The bench command hands the name and what follows `--` to the service; no name runs every program."""
    programs = commands.root / K_RELEASE_TREE / "tests"
    programs.mkdir()
    (programs / "benchmark_gemm").write_text("")
    result = commands.invoke(*args)
    assert result.exit_code == 0, result.output
    assert _argv_below(commands) == expected


@pytest.mark.parametrize("args", [["bench", "--", "--runs", "3"], ["bench", "--sortt"]])
def test_bench_command_rejects_options_that_belong_to_no_benchmark(commands, args):
    """Options need a benchmark name before the separator; an unknown flag is a usage error."""
    result = commands.invoke(*args)
    assert result.exit_code == K_EXIT_USAGE
    assert commands.runner.commands == []
