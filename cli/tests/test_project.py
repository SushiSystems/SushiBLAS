# test_project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the project service and the commands that print its results."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from command_harness import (
    K_DEBUG_TREE,
    K_EXECUTABLE,
    K_RELEASE_TREE,
    StubBuildEnv,
    StubConfig,
    StubIndex,
)
from sushicore.build_admission import (
    BuildStoppedError,
    compiler_cache,
    hardware_threads,
    launcher_defines,
    stop_file,
)
from sushicore.errors import ConfigError
from sushicore.console_memory import RecordingConsole
from sushicore.proc import Runner
from sushicore.proc_memory import RecordingRunner
from sushicore.testing import TestLabel

from sushiblas.services import project
from sushiblas.services.build_type import K_DEFAULT_BUILD_TYPE, BuildType
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
from sushiblas.services.tree_choice import TreeChoice

pytestmark = pytest.mark.medium

K_CONFIGURE_FAILED = 9
K_CHILD_EXIT = 7
K_INTERRUPTED = 130
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


@pytest.fixture
def checkout(commands):
    """Returns the scratch checkout of the commands fixture."""
    return commands.root


@pytest.fixture
def console():
    """Returns the console that records every line the service prints."""
    return RecordingConsole()


@pytest.fixture
def runner(console):
    """Returns the runner that records each command and starts no process."""
    return RecordingRunner(console)


@pytest.fixture
def parts(checkout, console, runner):
    """Returns the recording console and the stub driver and the service built on them."""
    driver = _Driver()
    service = ProjectService(
        console=console,
        runner=runner,
        driver=driver,
        index=StubIndex(),
        build_env=StubBuildEnv(),
        find_root=lambda: checkout,
        load_config=StubConfig,
    )
    return console, driver, service


def test_build_says_it_replaces_a_tree_of_another_generator(parts, checkout):
    """A tree the driver cannot reuse is removed out loud, not silently."""
    console, driver, service = parts
    with pytest.raises(StepFailedError) as raised:
        service.build(BuildType.DEBUG)
    assert raised.value.exit_code == K_CONFIGURE_FAILED
    assert not (checkout / K_DEBUG_TREE).exists()
    said = console.texts("info")
    reason = next(text for text in said if "empty tree" in text)
    assert said.index(reason) < said.index(f"Removing {checkout / K_DEBUG_TREE}...")


def test_build_removes_nothing_when_there_is_no_tree(parts, checkout):
    """A first build has no tree to announce or remove."""
    console, driver, service = parts
    (checkout / K_DEBUG_TREE).rmdir()
    with pytest.raises(StepFailedError):
        service.build(BuildType.DEBUG)
    assert not [text for text in console.texts("info") if "empty tree" in text]
    assert not [text for text in console.texts("info") if text.startswith("Removing")]


def test_build_rejects_a_bad_define_before_touching_the_tree(parts, checkout):
    """A -D without '=' raises with exit 2 and leaves the tree and its files in place."""
    console, driver, service = parts
    marker = checkout / K_DEBUG_TREE / "CMakeCache.txt"
    marker.write_text("")
    with pytest.raises(InvalidDefineError) as raised:
        service.build(BuildType.DEBUG, clean=True, defines=["SB_ENABLE_LOGGING=OFF", "NOVALUE"])
    assert raised.value.exit_code == K_EXIT_USAGE
    assert marker.is_file()
    assert driver.configured == []
    assert console.calls == []


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
        (None, ["-DSB_SANITIZE="]),
    ],
)
def test_build_passes_the_sanitizer_set_as_a_cache_variable(parts, checkout, sanitizer, expected):
    """--sanitize becomes one SB_SANITIZE flag; the plain tree, left out or none, clears it."""
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
        """Records the jobs the compile was asked for, leaves the tree behind and succeeds."""
        self.compiled_jobs.append(options.get("jobs"))
        build_dir.mkdir(parents=True, exist_ok=True)
        return 0


def _compiling_service(checkout) -> tuple[RecordingConsole, _CompilingDriver, ProjectService]:
    """Returns the console and driver that configure and compile, and the service on them."""
    console = RecordingConsole()
    driver = _CompilingDriver()
    service = ProjectService(
        console=console,
        runner=RecordingRunner(console),
        driver=driver,
        index=StubIndex(),
        build_env=StubBuildEnv(),
        find_root=lambda: checkout,
        load_config=StubConfig,
    )
    return console, driver, service


def test_build_hands_the_jobs_to_the_compile(checkout):
    """The jobs given reach the driver as given, and one per hardware thread when not given."""
    console, driver, service = _compiling_service(checkout)
    assert service.build(BuildType.RELEASE, jobs=4) == 0
    assert service.build(BuildType.RELEASE) == 0
    assert driver.compiled_jobs == [4, hardware_threads()]


def test_the_default_build_type_is_debug():
    """Every command that finds a tree and is not a benchmark starts from Debug."""
    assert K_DEFAULT_BUILD_TYPE is BuildType.DEBUG
    assert TreeChoice().build_type is BuildType.DEBUG


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        (TreeChoice(), Path("build") / "default-debug"),
        (TreeChoice(BuildType.RELEASE), Path("build") / "default-release"),
        (TreeChoice(BuildType.RELWITHDEBINFO), Path("build") / "default-relwithdebinfo"),
        (TreeChoice(BuildType.DEBUG, Sanitizer.ADDRESS_UNDEFINED), Path("build") / "asan-debug"),
        (TreeChoice(BuildType.RELEASE, Sanitizer.THREAD), Path("build") / "tsan-release"),
        (TreeChoice(BuildType.DEBUG, Sanitizer.NONE), Path("build") / "default-debug"),
    ],
)
def test_the_tree_directory_follows_the_type_and_the_sanitizer(tmp_path, tree, expected):
    """Each build type and sanitizer set has its own directory below build/."""
    assert tree.directory(tmp_path) == tmp_path / expected


def test_the_build_directory_override_names_the_tree(tmp_path):
    """--build-dir names the tree below the checkout whatever the type is."""
    assert TreeChoice().directory(tmp_path, "build/pkg-x") == tmp_path / "build" / "pkg-x"


@pytest.mark.parametrize(
    ("build_type", "expected"),
    [(BuildType.DEBUG, K_DEBUG_TREE), (BuildType.RELEASE, K_RELEASE_TREE)],
)
def test_build_configures_the_tree_of_its_type(checkout, build_type, expected):
    """The configure line names the tree of the type and its CMake spelling."""
    console, driver, service = _compiling_service(checkout)
    service.build(build_type)
    configured = driver.configured[0]
    assert configured[configured.index("-B") + 1] == str(checkout / expected)
    assert f"-DCMAKE_BUILD_TYPE={build_type.name.capitalize()}" in configured


def test_build_refuses_to_configure_a_tree_of_another_type(checkout):
    """A tree configured as Release is not configured as Debug, and nothing is removed."""
    console, driver, service = _compiling_service(checkout)
    tree = checkout / K_DEBUG_TREE
    (tree / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
    with pytest.raises(ConfigError):
        service.build(BuildType.DEBUG)
    assert (tree / "CMakeCache.txt").is_file()
    assert driver.configured == []


def test_build_refuses_a_define_that_sets_the_build_type(checkout):
    """-D CMAKE_BUILD_TYPE would split the tree from its name, so it is refused."""
    console, driver, service = _compiling_service(checkout)
    with pytest.raises(ConfigError):
        service.build(BuildType.DEBUG, defines=["CMAKE_BUILD_TYPE=Release"])
    assert driver.configured == []


def test_build_refuses_while_the_memory_gate_is_stopped(checkout):
    """The stop file of the gate ends the build before it configures."""
    console, driver, service = _compiling_service(checkout)
    stop = stop_file(checkout / K_DEBUG_TREE)
    stop.parent.mkdir(parents=True)
    stop.write_text("")
    with pytest.raises(BuildStoppedError):
        service.build(BuildType.DEBUG)
    assert driver.configured == []


def test_build_configure_carries_both_gate_launchers(checkout):
    """The configure line sets the compiler and the linker launcher of the memory gate."""
    console, driver, service = _compiling_service(checkout)
    service.build(BuildType.DEBUG)
    tree = checkout / K_DEBUG_TREE
    entries = launcher_defines(tree, compiler_cache(""))
    assert len(entries) == 2
    for entry in entries:
        assert f"-D{entry}" in driver.configured[0]


def test_build_prints_and_writes_the_report_on_request(checkout):
    """--report prints the report lines and writes build_report.json into the tree."""
    console, driver, service = _compiling_service(checkout)
    service.build(BuildType.DEBUG)
    assert not [text for text in console.texts("info") if text.startswith("Report:")]
    service.build(BuildType.DEBUG, report=True)
    written = checkout / K_DEBUG_TREE / "build_report.json"
    assert written.is_file()
    assert f"Report: {written}" in console.texts("info")


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


def test_test_cli_runs_pytest_with_the_marker_expression(parts, checkout, runner):
    """--cli runs pytest from cli/ with the expression of the labels given."""
    console, driver, service = parts
    assert service.test_cli([TestLabel.SMALL]) == 0
    assert runner.commands[0].argv[1:] == (
        "-m", "pytest", "-q", "-m", "(small) and not quarantined")
    assert runner.commands[0].cwd == checkout / "cli"


def _configure_of(commands) -> tuple[str, ...]:
    """Returns the cmake configure line the service handed the runner first."""
    return commands.runner.commands[0].argv


def _compile_of(commands) -> tuple[str, ...]:
    """Returns the cmake build line the service handed the runner second."""
    return commands.runner.commands[1].argv


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["build", "--benchmarks"], ["-DSB_BUILD_BENCHMARKS=ON"]),
        (["build", "--no-benchmarks"], ["-DSB_BUILD_BENCHMARKS=OFF"]),
        (["build"], []),
    ],
)
def test_build_command_forwards_the_benchmarks_option(commands, args, expected):
    """The build command hands the tri-state option to the service unchanged."""
    result = commands.invoke(*args)
    assert result.exit_code == 0, result.output
    flags = [word for word in _configure_of(commands) if word.startswith("-DSB_BUILD_BENCHMARKS")]
    assert flags == expected


def test_build_command_defaults_to_debug_and_forwards_the_report_option(commands):
    """With no option the build command asks for Debug and no report; --report asks for one."""
    assert commands.invoke("build").exit_code == 0
    assert "-DCMAKE_BUILD_TYPE=Debug" in _configure_of(commands)
    assert not [text for text in commands.console.texts("info") if text.startswith("Report:")]
    assert commands.invoke("build", "--report", "--type", "release").exit_code == 0
    assert "-DCMAKE_BUILD_TYPE=Release" in commands.runner.commands[2].argv
    assert f"Report: {commands.root / K_RELEASE_TREE / 'build_report.json'}" in (
        commands.console.texts("info"))


@pytest.mark.parametrize(
    ("args", "expected"),
    [(["build", "--jobs", "4"], "4"), (["build"], str(hardware_threads()))],
)
def test_build_command_forwards_the_jobs_option(commands, args, expected):
    """The build command hands --jobs to the service unchanged, and one per hardware thread when it is left out."""
    result = commands.invoke(*args)
    assert result.exit_code == 0, result.output
    build = _compile_of(commands)
    assert build[build.index("--parallel") + 1] == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["build", "--distributed"], ["-DSB_ENABLE_DISTRIBUTED=ON"]),
        (["build", "--no-distributed"], ["-DSB_ENABLE_DISTRIBUTED=OFF"]),
        (["build"], []),
    ],
)
def test_build_command_forwards_the_distributed_option(commands, args, expected):
    """The build command hands the tri-state --distributed option to the service unchanged."""
    result = commands.invoke(*args)
    assert result.exit_code == 0, result.output
    flags = [word for word in _configure_of(commands) if word.startswith("-DSB_ENABLE_DISTRIBUTED")]
    assert flags == expected


def test_distributed_flags_switch_the_cluster_module():
    """distributed_flags gives the SB_ENABLE_DISTRIBUTED define, or nothing for None."""
    assert project.distributed_flags(True) == ["-DSB_ENABLE_DISTRIBUTED=ON"]
    assert project.distributed_flags(False) == ["-DSB_ENABLE_DISTRIBUTED=OFF"]
    assert project.distributed_flags(None) == []


def test_clean_reports_a_removed_tree(parts, checkout):
    """clean returns the tree and True when it removed one."""
    console, driver, service = parts
    assert service.clean() == CleanResult(checkout / K_DEBUG_TREE, True)
    assert not (checkout / K_DEBUG_TREE).exists()
    assert (checkout / K_RELEASE_TREE).is_dir()


def test_clean_reports_a_missing_tree(parts, checkout):
    """clean returns False, not an error, when there is no tree."""
    console, driver, service = parts
    (checkout / K_DEBUG_TREE).rmdir()
    assert service.clean() == CleanResult(checkout / K_DEBUG_TREE, False)


def test_clean_raises_when_the_tree_remains(parts, checkout, monkeypatch):
    """A tree the removal could not delete raises instead of reporting success."""
    console, driver, service = parts
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


def test_run_returns_the_exit_code_of_the_child(parts, checkout, runner):
    """run launches the default target with the forwarded arguments and returns its code."""
    console, driver, service = parts
    exe = checkout / K_DEBUG_TREE / K_EXECUTABLE
    exe.write_text("")
    runner.script(K_CHILD_EXIT)
    assert service.run(app_args=["--flag"]) == K_CHILD_EXIT
    assert [command.argv for command in runner.commands] == [(str(exe), "--flag")]


def test_run_raises_for_a_target_that_matches_nothing(parts):
    """An unknown target raises instead of returning a bare 1."""
    console, driver, service = parts
    with pytest.raises(ExecutableNotFoundError):
        service.run(target="no_such_binary")


def test_run_raises_without_a_build_tree(parts, checkout):
    """run needs the build tree it is asked for, and says which command makes it."""
    console, driver, service = parts
    (checkout / K_DEBUG_TREE).rmdir()
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

    console = RecordingConsole()
    runner = Runner(console, "sb", catch_interrupt=True)
    monkeypatch.setattr(subprocess, "run", interrupt)
    assert runner.run(["some-tool"], tmp_path, {}) == K_INTERRUPTED
    assert console.texts("warn") == ["Interrupted."]


def test_bench_raises_when_no_benchmark_program_is_built(parts, checkout):
    """bench with no benchmark program says how to build one."""
    console, driver, service = parts
    with pytest.raises(ExecutableNotFoundError):
        service.bench()


def test_bench_runs_every_benchmark_program_in_name_order(parts, checkout, runner):
    """bench runs each benchmark_* program under build/tests, skipping the test programs and the ones that take arguments."""
    console, driver, service = parts
    programs = checkout / K_RELEASE_TREE / "tests"
    programs.mkdir()
    for name in ("benchmark_sparse", "benchmark_gemm", "benchmark_gemm_blocking", "sushiblas_operations_tests"):
        (programs / name).write_text("")
    assert service.bench() == 0
    assert [Path(command.argv[0]).name for command in runner.commands] == [
        "benchmark_gemm", "benchmark_sparse"]


def test_bench_runs_the_named_scenario_benchmark_with_the_forwarded_options(
    parts, checkout, runner
):
    """bench with a name runs that one executable and passes the options after it."""
    console, driver, service = parts
    exe = checkout / K_RELEASE_TREE / K_EXECUTABLE
    exe.write_text("")
    runner.script(K_CHILD_EXIT)
    assert service.bench(K_EXECUTABLE, ["--out", "folder"]) == K_CHILD_EXIT
    assert [command.argv for command in runner.commands] == [(str(exe), "--out", "folder")]


def test_bench_raises_for_a_scenario_benchmark_that_matches_nothing(parts, checkout):
    """bench with a name that matches no executable raises instead of running every program."""
    console, driver, service = parts
    with pytest.raises(ExecutableNotFoundError):
        service.bench("no_such_benchmark")


def test_bench_finds_the_release_tree_unless_told_another(parts, checkout):
    """A benchmark measures the build that ships: its default tree is the Release one."""
    console, driver, service = parts
    (checkout / K_RELEASE_TREE).rmdir()
    with pytest.raises(BuildTreeMissingError) as raised:
        service.bench()
    assert "sb build --type release" in str(raised.value)


def test_bench_command_forwards_the_name_and_the_options_after_the_separator(commands):
    """The bench command hands the name and what follows `--` to the service."""
    exe = commands.root / K_RELEASE_TREE / K_EXECUTABLE
    exe.write_text("")
    result = commands.invoke("bench", K_EXECUTABLE, "--", "--runs", "3")
    assert result.exit_code == 0, result.output
    assert [command.argv for command in commands.runner.commands] == [(str(exe), "--runs", "3")]


def test_bench_command_without_a_name_runs_every_program(commands):
    """The bench command with no name hands the service no name and no options."""
    programs = commands.root / K_RELEASE_TREE / "tests"
    programs.mkdir()
    (programs / "benchmark_gemm").write_text("")
    result = commands.invoke("bench")
    assert result.exit_code == 0, result.output
    assert [command.argv for command in commands.runner.commands] == [
        (str(programs / "benchmark_gemm"),)]


@pytest.mark.parametrize("args", [["bench", "--", "--runs", "3"], ["bench", "--sortt"]])
def test_bench_command_rejects_options_that_belong_to_no_benchmark(commands, args):
    """Options need a benchmark name before the separator; an unknown flag is a usage error."""
    result = commands.invoke(*args)
    assert result.exit_code == K_EXIT_USAGE
    assert commands.runner.commands == []
