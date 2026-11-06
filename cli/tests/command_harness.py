# command_harness.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Builds the real project service over a scratch checkout, a recording runner and a recording console.

A command test hands the context this module builds to ``command_context.using`` and reads
what the service handed cmake, ctest and the programs from ``runner.commands``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sushicore.cmake_driver import CMakeDriver
from sushicore.console_memory import RecordingConsole
from sushicore.proc_memory import RecordingRunner
from typer.testing import CliRunner, Result

from sushiblas import cli, command_context
from sushiblas.command_context import CommandContext
from sushiblas.services.project import ProjectService

K_EXECUTABLE = "quickstart_demo"
#: The trees a plain build and a benchmark find, below the checkout.
K_DEBUG_TREE = Path("build") / "default-debug"
K_RELEASE_TREE = Path("build") / "default-release"


class StubIndex:
    """Finds one executable, named K_EXECUTABLE, directly under the build tree."""

    def match(self, build_root: Path, query: str) -> Path | None:
        """Returns the executable when *query* names it."""
        exe = build_root / K_EXECUTABLE
        return exe if query == K_EXECUTABLE and exe.is_file() else None


class StubBuildEnv:
    """Returns an empty environment without running a shell."""

    def load(self, cfg, build_dir: Path) -> dict[str, str]:
        """Returns no variables."""
        return {}


class StubPackages:
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


class StubConfig:
    """Holds the few settings the service and the CMake driver read."""

    generator = "Ninja"
    ccache_exe = ""
    cmake_exe = ""
    ctest_exe = ""
    doxygen_exe = ""
    is_windows = False
    platform = "linux"
    target_bin = K_EXECUTABLE

    def expand(self, value: str) -> str:
        """Returns *value* as it is; no variable is set."""
        return value

    def sushi_definitions(self, root: Path, packaged=None) -> list[str]:
        """Returns no definition: every pinned library is a package."""
        return []

    def sushi_packages(self, root: Path, build_type: str, options=None, report=print,
                       warn=print, compiler: str = ""):
        """Returns a package set with no library."""
        return StubPackages()

    def resolved_compiler(self, root: Path) -> str:
        """Returns a compiler name; nothing runs it."""
        return "clang++"

    def resolved_vcpkg(self, root: Path) -> str:
        """Returns no vcpkg tree."""
        return ""


@dataclass(frozen=True, slots=True)
class Commands:
    """The real service over a scratch checkout, with what it printed and ran kept for reading.

    Args:
        root: The scratch checkout.
        runner: Records each command the service runs and answers the scripted results.
        console: Records each line the commands and the service print.
        context: The object a command reads from Typer's context.
    """

    root: Path
    runner: RecordingRunner
    console: RecordingConsole
    context: CommandContext

    def invoke(self, *arguments: str) -> Result:
        """Runs ``sb *arguments*`` on this context and returns Typer's result."""
        with command_context.using(self.context):
            return CliRunner().invoke(cli.app, list(arguments))


def make_service(root: Path, config: StubConfig | None = None
                 ) -> tuple[RecordingRunner, RecordingConsole, ProjectService]:
    """Returns a recording runner and console and the real service, with its real CMake driver, on them."""
    console = RecordingConsole()
    runner = RecordingRunner(console)
    settings = config or StubConfig()
    service = ProjectService(
        console=console,
        runner=runner,
        driver=CMakeDriver(console, runner),
        index=StubIndex(),
        build_env=StubBuildEnv(),
        find_root=lambda: root,
        load_config=lambda: settings,
    )
    return runner, console, service


def make_commands(root: Path, config: StubConfig | None = None) -> Commands:
    """Returns the commands over the checkout *root*, whose service uses *config* when given."""
    runner, console, service = make_service(root, config)
    context = CommandContext(
        service_factory=lambda: service, console=console, project_root=lambda: root)
    return Commands(root, runner, console, context)
