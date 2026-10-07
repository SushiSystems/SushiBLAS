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
from sushiblas.services.device import K_DEVICE_VARIABLE
from sushiblas.services.errors import (
    K_EXIT_USAGE,
    BuildTreeMissingError,
    ExecutableNotFoundError,
    InvalidDefineError,
    InvalidDeviceError,
    StepFailedError,
    SuiteOptionError,
    TreeNotRemovedError,
)
from sushiblas.services.project import CleanResult, ProjectService
from sushiblas.services.suite import Suite

K_CONFIGURE_FAILED = 9
K_CHILD_EXIT = 7
K_INTERRUPTED = 130
K_EXECUTABLE = "sushiblas_example"
K_CALLER_SELECTOR = "level_zero:0"


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
    """Stands in for the CMake driver: reports a tree to replace and fails to configure."""

    def __init__(self) -> None:
        self.configured: list[list[str]] = []
        self.ctest_environments: list[dict[str, str]] = []

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

    def ctest_run(self, cfg, build_dir: Path, env, **_) -> int:
        """Records the environment ctest would run under and reports success."""
        self.ctest_environments.append(dict(env))
        return 0


class _Index:
    """Finds one executable, named K_EXECUTABLE, directly under the build tree."""

    def match(self, build_root: Path, query: str) -> Path | None:
        """Returns the executable when *query* names it."""
        exe = build_root / K_EXECUTABLE
        return exe if query == K_EXECUTABLE and exe.is_file() else None


class _BuildEnv:
    """Returns a fixed environment, empty unless a test fills it, without running a shell."""

    def __init__(self) -> None:
        self.variables: dict[str, str] = {}

    def load(self, cfg, build_dir: Path) -> dict[str, str]:
        """Returns a copy of the variables the test set."""
        return dict(self.variables)


class _Config:
    """Holds the few settings the service reads."""

    generator = "Ninja"
    is_windows = False
    target_bin = K_EXECUTABLE

    def runtime_dir(self, root: Path) -> Path:
        """Returns the sibling checkout the test wrote."""
        return root.parent / "sushiruntime"

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


@pytest.fixture
def device_parts(checkout):
    """Returns the console, driver and build environment and the service built on them."""
    console = _Console()
    driver = _Driver()
    build_env = _BuildEnv()
    service = ProjectService(
        console=console,
        runner=Runner(_Console(), "sb", catch_interrupt=True),
        driver=driver,
        index=_Index(),
        build_env=build_env,
        find_root=lambda: checkout,
        load_config=_Config,
    )
    return console, driver, build_env, service


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
        service.build(BuildType.DEBUG, clean=True, defines=["SB_LOG_LEVEL=INFO", "NOVALUE"])
    assert raised.value.exit_code == K_EXIT_USAGE
    assert marker.is_file()
    assert driver.configured == []
    assert console.lines == []


def test_build_appends_defines_after_every_default(parts, checkout):
    """Caller -D flags end the configure line, so no default outranks them."""
    console, driver, service = parts
    with pytest.raises(StepFailedError):
        service.build(BuildType.RELEASE, defines=["SB_LOG_LEVEL=INFO", "SB_NATIVE_ARCH="])
    assert driver.configured[0][-2:] == ["-DSB_LOG_LEVEL=INFO", "-DSB_NATIVE_ARCH="]


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
        def build(self, build_type, clean, define, benchmarks):
            received.append(benchmarks)
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0
    assert received == [expected]


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


def _child_environments(monkeypatch) -> list[dict[str, str]]:
    """Makes every child process exit 0 and returns the list of environments they ran under."""
    environments: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        """Records the environment of *cmd* and reports success."""
        environments.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sushicore.proc.subprocess, "run", fake_run)
    return environments


def test_test_leaves_the_environment_alone_without_a_device(device_parts):
    """No --device: ctest runs under what the build environment held, selector included."""
    console, driver, build_env, service = device_parts
    build_env.variables = {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}
    assert service.test(Suite.UNIT) == 0
    assert driver.ctest_environments == [{K_DEVICE_VARIABLE: K_CALLER_SELECTOR}]
    assert console.lines == []


def test_test_sets_no_selector_without_a_device_or_a_variable(device_parts):
    """No --device and no variable: the child environment carries no selector."""
    console, driver, build_env, service = device_parts
    assert service.test(Suite.UNIT) == 0
    assert driver.ctest_environments == [{}]


def test_test_runs_ctest_under_the_selector_of_the_device(device_parts):
    """--device puts its selector in ctest's environment and says nothing when none was set."""
    console, driver, build_env, service = device_parts
    assert service.test(Suite.UNIT, device="cpu") == 0
    assert driver.ctest_environments == [{K_DEVICE_VARIABLE: "opencl:cpu"}]
    assert console.lines == []


def test_test_says_once_that_the_device_outranks_the_environment(device_parts):
    """--device wins over a selector in the environment, and one line names both values."""
    console, driver, build_env, service = device_parts
    build_env.variables = {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}
    assert service.test(Suite.UNIT, device="cuda") == 0
    assert driver.ctest_environments == [{K_DEVICE_VARIABLE: "cuda:*"}]
    said = console.texts("info")
    assert len(said) == 1
    assert K_CALLER_SELECTOR in said[0] and "cuda:*" in said[0]


def test_test_rejects_a_bad_device_before_anything_runs(device_parts, checkout):
    """An unknown --device raises with exit 2, even where there is no build tree to report."""
    console, driver, build_env, service = device_parts
    (checkout / "build").rmdir()
    with pytest.raises(InvalidDeviceError) as raised:
        service.test(Suite.UNIT, device="gpu")
    assert raised.value.exit_code == K_EXIT_USAGE
    assert driver.ctest_environments == []


def test_run_starts_the_program_under_the_selector_of_the_device(
    device_parts, checkout, monkeypatch
):
    """run passes the raw selector to the child and reports the one it replaced."""
    console, driver, build_env, service = device_parts
    build_env.variables = {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}
    (checkout / "build" / K_EXECUTABLE).write_text("")
    environments = _child_environments(monkeypatch)
    assert service.run(device="selector:level_zero:gpu") == 0
    assert environments == [{K_DEVICE_VARIABLE: "level_zero:gpu"}]
    overrides = [text for text in console.texts("info") if "--device overrides" in text]
    assert len(overrides) == 1


def test_run_leaves_the_environment_alone_without_a_device(device_parts, checkout, monkeypatch):
    """No --device: the program runs under the selector the caller set."""
    console, driver, build_env, service = device_parts
    build_env.variables = {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}
    (checkout / "build" / K_EXECUTABLE).write_text("")
    environments = _child_environments(monkeypatch)
    assert service.run() == 0
    assert environments == [{K_DEVICE_VARIABLE: K_CALLER_SELECTOR}]
    assert not [text for text in console.texts("info") if "--device" in text]


def test_run_rejects_a_bad_device_before_it_looks_for_the_program(device_parts):
    """An unknown --device raises with exit 2 before the missing executable is noticed."""
    console, driver, build_env, service = device_parts
    with pytest.raises(InvalidDeviceError):
        service.run(device="selector:")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["test", "--device", "cpu"], "cpu"),
        (["test"], None),
        (["run", "--device", "selector:cuda:0"], "selector:cuda:0"),
        (["run", "bench_gemm", "--device", "cuda", "--", "--size", "64"], "cuda"),
        (["run"], None),
    ],
)
def test_commands_forward_the_device_option(monkeypatch, args, expected):
    """test and run hand --device to the service as typed, and None when it is left out."""
    received = []

    class _Service:
        def test(self, suite, filter, repeat, device):
            received.append(device)
            return 0

        def run(self, target, sort, app_args, device):
            received.append(device)
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0
    assert received == [expected]


@pytest.mark.parametrize("command", ["test", "run"])
def test_commands_exit_two_on_a_bad_device(device_parts, monkeypatch, command):
    """A --device that names nothing prints one error line and exits 2."""
    console, driver, build_env, service = device_parts
    monkeypatch.setattr(cli, "_service", lambda: service)
    printed = _Console()
    monkeypatch.setattr(cli, "console", printed)
    result = CliRunner().invoke(cli.app, [command, "--device", "gpu"])
    assert result.exit_code == K_EXIT_USAGE
    assert len(printed.texts("error")) == 1
    assert "'gpu'" in printed.texts("error")[0]


def test_package_suite_refuses_ctest_options(parts):
    """--filter with the package suite raises with exit 2."""
    console, driver, service = parts
    with pytest.raises(SuiteOptionError) as raised:
        service.test(Suite.PACKAGE, filter="Gemm.*")
    assert raised.value.exit_code == K_EXIT_USAGE


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
