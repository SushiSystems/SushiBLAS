# test_library_packages.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that `sb` consumes SushiRuntime and SushiLog as packages and honours the package contract."""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python below 3.11
    import tomli as tomllib

import pytest
from sushicore import deps_fragment
from sushicore.proc import Runner
from typer.testing import CliRunner

from sushiblas import cli, config
from sushiblas.config import Config
from sushiblas.services import library_packages
from sushiblas.services.build_type import BuildType
from sushiblas.services.errors import LeanBenchmarksError
from sushiblas.services.project import ProjectService

pytestmark = pytest.mark.medium

K_RUNTIME_CONFIG = "/store/sushiruntime/abc/prefix/lib/cmake/SushiRuntime"
K_LOG_CONFIG = "/store/sushilog/def/prefix/lib/cmake/SushiLog"
K_PROFILER_CONFIG = "/store/sushiprofiler/ghi/prefix/lib/cmake/SushiProfiler"
K_RUN_DIRS = ["/store/sushiruntime/abc/prefix/lib", "/store/sushilog/def/prefix/lib",
              "/store/sushiprofiler/ghi/prefix/lib"]
K_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class _Console:
    """Swallows every line the service prints."""

    def __init__(self) -> None:
        self.console = self

    def __getattr__(self, kind: str):
        return lambda *arguments, **options: None


class _Packages:
    """Stands in for the resolved packages of SushiRuntime, SushiLog and SushiProfiler."""

    def names(self) -> list[str]:
        """Returns the library names."""
        return ["sushilog", "sushiprofiler", "sushiruntime"]

    def definitions(self) -> list[str]:
        """Returns the config folder definitions."""
        return [f"-DSushiLog_DIR={K_LOG_CONFIG}", f"-DSushiProfiler_DIR={K_PROFILER_CONFIG}",
                f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}"]

    def run_dirs(self, platform: str) -> list[str]:
        """Returns the run-time folders."""
        return list(K_RUN_DIRS)


class _Config:
    """Holds the few settings the service reads and records what it asks the resolver."""

    generator = "Ninja"
    is_windows = False
    platform = "linux"
    target_bin = "quickstart_demo"

    def __init__(self) -> None:
        self.asked: list[dict] = []

    def sushi_definitions(self, root: Path, packaged=None) -> list[str]:
        """Returns a checkout definition for a pin that is not packaged; there is none."""
        return []

    def sushi_packages(self, root, build_type, options=None, report=print, warn=print,
                       compiler=""):
        """Records the request and returns the two packages."""
        self.asked.append({"build_type": build_type, "options": dict(options or {}),
                           "compiler": compiler})
        return _Packages()

    def resolved_compiler(self, root: Path) -> str:
        """Returns a compiler name; nothing runs it."""
        return "clang++"

    def resolved_vcpkg(self, root: Path) -> str:
        """Returns no vcpkg tree."""
        return ""


class _Driver:
    """Records the configure and compile calls and the environment of each."""

    def __init__(self) -> None:
        self.configured: list[list[str]] = []
        self.environments: list[dict[str, str]] = []

    def cmake(self, cfg) -> str:
        """Returns the cmake name."""
        return "cmake"

    def needs_configure(self, build_dir, generator, **_) -> bool:
        """Reports the tree as reusable."""
        return False

    def configure(self, args, cwd, env=None) -> int:
        """Records the arguments and the environment of one configure."""
        self.configured.append(list(args))
        self.environments.append(env)
        return 0

    def compile(self, cfg, build_dir, cwd, env, **options) -> int:
        """Succeeds."""
        return 0


class _BuildEnv:
    """Returns an environment that already has a library path."""

    def load(self, cfg, build_dir) -> dict[str, str]:
        """Returns one variable."""
        return {"LD_LIBRARY_PATH": "/old"}


@pytest.fixture
def parts(tmp_path):
    """Returns the service over a scratch checkout, with its config and driver."""
    root = tmp_path / "sushiblas"
    root.mkdir()
    cfg, driver, console = _Config(), _Driver(), _Console()
    service = ProjectService(
        console=console,
        runner=Runner(console, "sb", catch_interrupt=True),
        driver=driver,
        index=None,
        build_env=_BuildEnv(),
        find_root=lambda: root,
        load_config=lambda: cfg,
    )
    return root, cfg, driver, service


def test_the_build_asks_for_the_closure_with_its_compiler_and_type(parts):
    root, cfg, driver, service = parts

    assert service.build(BuildType.DEBUG) == 0

    assert cfg.asked == [{"build_type": "debug", "options": {}, "compiler": "clang++"}]


def test_a_distributed_build_builds_the_runtime_package_distributed(parts):
    root, cfg, driver, service = parts

    assert service.build(BuildType.RELEASE, distributed=True) == 0

    assert cfg.asked[0]["options"] == {"sushiruntime": ("--distributed",)}


def test_a_build_that_leaves_distributed_out_keeps_what_the_tree_has(parts):
    root, cfg, driver, service = parts
    (root / "build").mkdir()
    (root / "build" / "CMakeCache.txt").write_text("SB_ENABLE_DISTRIBUTED:BOOL=ON\n")

    assert service.build(BuildType.RELEASE) == 0

    assert cfg.asked[0]["options"] == {"sushiruntime": ("--distributed",)}


def test_the_configure_passes_each_package_and_no_checkout_directory(parts):
    root, cfg, driver, service = parts

    assert service.build(BuildType.RELEASE) == 0

    arguments = driver.configured[0]
    assert f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}" in arguments
    assert f"-DSushiLog_DIR={K_LOG_CONFIG}" in arguments
    assert f"-DSushiProfiler_DIR={K_PROFILER_CONFIG}" in arguments
    assert not [item for item in arguments
                if item.startswith(("-DSUSHIRUNTIME_DIR", "-DSUSHILOG_DIR",
                                          "-DSUSHIPROFILER_DIR"))]


def test_the_run_dirs_lead_the_search_path_of_the_build_and_of_a_later_run(parts):
    root, cfg, driver, service = parts

    assert service.build(BuildType.RELEASE) == 0

    assert driver.environments[0]["LD_LIBRARY_PATH"].split(os.pathsep) == [*K_RUN_DIRS, "/old"]
    assert library_packages.read_run_dirs(root / "build") == K_RUN_DIRS
    assert service._run_env(cfg, root / "build")["LD_LIBRARY_PATH"].split(os.pathsep) == [
        *K_RUN_DIRS, "/old"]


def test_the_windows_run_dirs_lead_path():
    env = {"PATH": "C:/old"}

    library_packages.prepend_run_dirs(Config(platform="windows"), env, ["C:/pkg/bin"])

    assert env["PATH"].startswith("C:/pkg/bin")


def test_a_lean_build_leaves_out_the_tests_the_examples_and_the_benchmarks(parts):
    root, cfg, driver, service = parts

    assert service.build(BuildType.RELEASE, lean=True, tree_path="build/pkg-x") == 0

    arguments = driver.configured[0]
    for switch in ("SB_BUILD_TESTS", "SB_BUILD_EXAMPLES", "SB_BUILD_BENCHMARKS"):
        assert f"-D{switch}=OFF" in arguments
    assert arguments[arguments.index("-B") + 1] == str(root / "build" / "pkg-x")
    assert library_packages.read_run_dirs(root / "build" / "pkg-x") == K_RUN_DIRS


def test_a_plain_build_asks_for_the_tests_and_the_examples(parts):
    root, cfg, driver, service = parts

    assert service.build(BuildType.RELEASE) == 0

    arguments = driver.configured[0]
    assert "-DSB_BUILD_TESTS=ON" in arguments and "-DSB_BUILD_EXAMPLES=ON" in arguments


def test_lean_refuses_the_benchmarks(parts):
    root, cfg, driver, service = parts

    with pytest.raises(LeanBenchmarksError):
        service.build(BuildType.RELEASE, lean=True, benchmarks=True)

    assert driver.configured == []


def test_the_build_command_forwards_the_tree_and_lean(monkeypatch):
    received: list[tuple] = []

    class _Service:
        def build(self, *arguments):
            received.append(arguments[-2:])
            return 0

    monkeypatch.setattr(cli, "_service", lambda: _Service())
    monkeypatch.setattr(cli, "console", _Console())

    result = CliRunner().invoke(cli.app, ["build", "--build-dir", "build/pkg-x", "--lean"])

    assert result.exit_code == 0
    assert received == [(True, "build/pkg-x")]


def test_the_package_command_is_registered_with_its_options():
    result = CliRunner().invoke(cli.app, ["package", "--help"])
    text = K_ANSI_ESCAPE.sub("", result.output)

    assert result.exit_code == 0
    for option in ("--build-dir", "--prefix", "--no-archive", "--force"):
        assert option in text


def test_the_compiler_the_environment_names_outranks_the_pinned_one(monkeypatch, tmp_path):
    cfg = Config(cxx="/pinned/clang++", stack_cxx="/stack/clang++")

    assert cfg.resolved_compiler(tmp_path) == "/stack/clang++"


def test_load_config_reads_the_stack_compiler_from_the_environment(monkeypatch):
    monkeypatch.setattr(config, "_MODULE", type("Module", (), {"load": lambda self, cls: cls()})())
    monkeypatch.setenv(config.K_STACK_CXX_ENV, "/env/clang++")

    assert config.load_config().stack_cxx == "/env/clang++"


def test_the_fragment_names_the_cli():
    fragment = deps_fragment.read(Path(__file__).resolve().parents[1] / "sushistack.deps.toml")

    assert fragment.cli == "sb"


def test_the_dependency_fragment_names_the_three_libraries_of_the_closure():
    fragment = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "sushistack.deps.toml").read_text(encoding="utf-8"))

    assert set(fragment["module"]["depends_on"]) == {"sushiruntime", "sushilog", "sushiprofiler"}
    assert fragment["sushisystems"]["sushiprofiler"]["version"] == "v0.3.2"
