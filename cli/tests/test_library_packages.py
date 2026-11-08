# test_library_packages.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that `sb` consumes SushiRuntime, SushiLog and SushiProfiler as packages and honours the package contract."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python below 3.11
    import tomli as tomllib

import pytest
from command_harness import K_EXECUTABLE, make_service
from sushicore import deps_fragment
from typer.testing import CliRunner

from sushiblas import cli, config
from sushiblas.config import Config
from sushiblas.services import library_packages
from sushiblas.services.build_type import BuildType
from sushiblas.services.package_source_memory import (
    MemoryPackageSet,
    MemoryPackageSource,
    PackageRequest,
)
from sushiblas.services.tree_choice import TreeChoice

K_FRAGMENT = Path(__file__).resolve().parents[1] / "sushistack.deps.toml"
K_LIBRARIES = ("sushiruntime", "sushilog", "sushiprofiler")
K_RUNTIME_CONFIG = "/store/sushiruntime/abc/prefix/lib/cmake/SushiRuntime"
K_LOG_CONFIG = "/store/sushilog/def/prefix/lib/cmake/SushiLog"
K_PROFILER_CONFIG = "/store/sushiprofiler/ghi/prefix/lib/cmake/SushiProfiler"
K_RUN_DIRS = ["/store/sushiruntime/abc/prefix/lib", "/store/sushilog/def/prefix/lib",
              "/store/sushiprofiler/ghi/prefix/lib"]
K_PACKAGES = MemoryPackageSet(
    library_names=("sushilog", "sushiprofiler", "sushiruntime"),
    cmake_definitions=(f"-DSushiLog_DIR={K_LOG_CONFIG}", f"-DSushiProfiler_DIR={K_PROFILER_CONFIG}",
                       f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}"),
    folders=tuple(K_RUN_DIRS),
)
K_OLD_PATH = "/old"
K_LIBRARY_PATH = "LD_LIBRARY_PATH"
K_CHECKOUT_DIRECTORIES = ("-DSUSHIRUNTIME_DIR", "-DSUSHILOG_DIR", "-DSUSHIPROFILER_DIR")
K_PACKAGE_TREE = "build/pkg-x"
K_PACKAGE_OPTIONS = ("--build-dir", "--prefix", "--no-archive", "--force")
K_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def checkout(tmp_path) -> Path:
    """Returns a scratch checkout that pins the three libraries, with a checkout of each beside it."""
    root = tmp_path / "sushiblas"
    (root / "cli").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("")
    shutil.copyfile(K_FRAGMENT, root / "cli" / "sushistack.deps.toml")
    (tmp_path / "sushiruntime").mkdir()
    (tmp_path / "sushilog").mkdir()
    (tmp_path / "sushiprofiler").mkdir()
    return root


@pytest.fixture
def source() -> MemoryPackageSource:
    """Returns the package source that answers with the three packages and keeps each request."""
    return MemoryPackageSource(K_PACKAGES)


@pytest.fixture
def built(checkout, source):
    """Returns the recording runner and the service over the checkout, whose environment has a library path."""
    runner, _, service = make_service(checkout, packages=source,
                                      environment={K_LIBRARY_PATH: K_OLD_PATH})
    return runner, service


def _search_path(env: dict[str, str]) -> list[str]:
    """Returns the folders of the library search path of *env*, in order."""
    return env[K_LIBRARY_PATH].split(os.pathsep)


def _absent_from(text: str, words: tuple[str, ...]) -> list[str]:
    """Returns the entries of *words* that *text* does not hold."""
    return [word for word in words if word not in text]


def _checkout_directories(argv: tuple[str, ...]) -> list[str]:
    """Returns the words of *argv* that name a library checkout instead of a package."""
    return [word for word in argv if word.startswith(K_CHECKOUT_DIRECTORIES)]


@pytest.mark.medium
class TestABuildLinksThePackages:
    """What `ProjectService.build` asks the package source for and hands cmake and the programs."""

    def test_a_build_asks_for_the_closure_of_its_type_with_its_compiler(self, built, source):
        """A Debug build asks once for the Release closure, with no options and the compiler it builds with."""
        runner, service = built
        assert service.build(BuildType.DEBUG) == 0
        assert source.requests == [PackageRequest("release", {}, "clang++")]

    def test_a_build_that_leaves_distributed_out_keeps_what_the_tree_has(self, built, source,
                                                                          checkout):
        """A tree cached with the cluster module asks for the distributed runtime when the build does not say."""
        runner, service = built
        tree = checkout / "build" / "default-release"
        tree.mkdir(parents=True)
        (tree / "CMakeCache.txt").write_text("SB_ENABLE_DISTRIBUTED:BOOL=ON\n")
        assert service.build(BuildType.RELEASE) == 0
        assert source.requests[0].options == {"sushiruntime": ("--distributed",)}

    def test_the_configure_passes_each_package_and_no_checkout_directory(self, built):
        """Each library reaches cmake as its package's config folder, never as the checkout beside it."""
        runner, service = built
        assert service.build(BuildType.RELEASE) == 0
        configured = runner.commands[0].argv
        assert f"-DSushiRuntime_DIR={K_RUNTIME_CONFIG}" in configured
        assert f"-DSushiLog_DIR={K_LOG_CONFIG}" in configured
        assert f"-DSushiProfiler_DIR={K_PROFILER_CONFIG}" in configured
        assert _checkout_directories(configured) == []

    def test_the_run_dirs_lead_the_search_path_of_the_build(self, built):
        """The configure runs with the package folders first on the library path, before the old ones."""
        runner, service = built
        assert service.build(BuildType.RELEASE) == 0
        assert _search_path(runner.commands[0].env) == [*K_RUN_DIRS, K_OLD_PATH]

    def test_a_build_records_the_run_dirs_in_its_tree(self, built, checkout):
        """The build writes the package folders into the tree it configured."""
        runner, service = built
        assert service.build(BuildType.RELEASE) == 0
        assert library_packages.read_run_dirs(checkout / "build" / "default-release") == K_RUN_DIRS

    def test_a_later_run_of_the_tree_puts_its_run_dirs_first(self, built, checkout):
        """A program of a tree runs with the folders the tree recorded first on the library path."""
        runner, service = built
        tree = checkout / "build" / "default-release"
        library_packages.record_run_dirs(tree, K_RUN_DIRS)
        assert service.run(K_EXECUTABLE, tree=TreeChoice(BuildType.RELEASE)) == 0
        assert _search_path(runner.commands[0].env) == [*K_RUN_DIRS, K_OLD_PATH]

    @pytest.mark.parametrize(
        ("lean", "expected"),
        [
            (True, ("-DSB_BUILD_TESTS=OFF", "-DSB_BUILD_EXAMPLES=OFF", "-DSB_BUILD_BENCHMARKS=OFF")),
            (False, ("-DSB_BUILD_TESTS=ON", "-DSB_BUILD_EXAMPLES=ON")),
        ],
    )
    def test_a_build_configures_what_its_tree_builds(self, built, lean, expected):
        """A lean build configures no tests, examples or benchmarks; a plain one asks for the tests and examples."""
        runner, service = built
        assert service.build(BuildType.RELEASE, lean=lean) == 0
        assert set(expected) <= set(runner.commands[0].argv)

    def test_the_build_command_forwards_the_tree_and_lean(self, commands):
        """`sb build --build-dir build/pkg-x --lean` configures that tree below the checkout as a lean one."""
        result = commands.invoke("build", "--build-dir", K_PACKAGE_TREE, "--lean")
        assert result.exit_code == 0, result.output
        configured = commands.runner.commands[0].argv
        assert "-DSB_BUILD_TESTS=OFF" in configured
        assert configured[configured.index("-B") + 1] == str(commands.root / "build" / "pkg-x")


@pytest.mark.medium
class TestTheConfigReadsTheStackCompiler:
    """What `load_config` takes from the environment over the layered files of a checkout."""

    def test_load_config_reads_the_stack_compiler_from_the_environment(self, checkout,
                                                                       monkeypatch):
        """SUSHISTACK_CXX becomes the stack compiler of the configuration loaded in a checkout."""
        (checkout / "cli" / "config.toml").write_text("")
        monkeypatch.chdir(checkout)
        monkeypatch.delenv("SUSHISTACK_HOME", raising=False)
        monkeypatch.setenv(config.K_STACK_CXX_ENV, "/env/clang++")
        assert config.load_config().stack_cxx == "/env/clang++"


@pytest.mark.medium
class TestTheDependencyFragment:
    """What `cli/sushistack.deps.toml` declares to the stack."""

    def test_the_fragment_names_the_cli(self):
        """The fragment names `sb` as the CLI of this module."""
        assert deps_fragment.read(K_FRAGMENT).cli == "sb"

    def test_the_fragment_depends_on_the_three_libraries_of_the_closure(self):
        """The module depends on SushiRuntime, SushiLog and SushiProfiler, and on nothing else."""
        fragment = tomllib.loads(K_FRAGMENT.read_text(encoding="utf-8"))
        assert set(fragment["module"]["depends_on"]) == set(K_LIBRARIES)

    @pytest.mark.parametrize("name", K_LIBRARIES)
    def test_the_fragment_pins_each_library_to_a_tag_and_a_full_commit(self, name):
        """Each pin names a v-tag and a 40-character commit."""
        pin = tomllib.loads(K_FRAGMENT.read_text(encoding="utf-8"))["sushisystems"][name]
        assert pin["version"].startswith("v")
        assert len(pin["commit"]) == 40


@pytest.mark.small
class TestThePackageOptionsAndTheSearchPath:
    """What the package command offers and what the search path and the compiler resolve to, in memory."""

    def test_the_package_command_is_registered_with_its_options(self):
        """`sb package --help` lists --build-dir, --prefix, --no-archive and --force."""
        result = CliRunner().invoke(cli.app, ["package", "--help"])
        text = K_ANSI_ESCAPE.sub("", result.output)
        assert result.exit_code == 0
        assert _absent_from(text, K_PACKAGE_OPTIONS) == []

    def test_the_windows_run_dirs_lead_path(self):
        """On Windows the package folders go first on PATH."""
        env = {"PATH": "C:/old"}
        library_packages.prepend_run_dirs(Config(platform="windows"), env, ["C:/pkg/bin"])
        assert env["PATH"].startswith("C:/pkg/bin")

    def test_the_stack_compiler_outranks_the_pinned_one(self):
        """A stack compiler set on the configuration wins over the configured cxx."""
        cfg = Config(cxx="/pinned/clang++", stack_cxx="/stack/clang++")
        assert cfg.resolved_compiler(Path("/checkout")) == "/stack/clang++"
