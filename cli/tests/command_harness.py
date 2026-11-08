# command_harness.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Builds the real project service over a scratch checkout, a recording runner and a recording console.

A command test hands the context built here to ``command_context.using`` and reads what the
service handed cmake, ctest and the programs from ``runner.commands``. See cli/tests/README.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sushicore.cmake_driver import CMakeDriver
from sushicore.console_memory import RecordingConsole
from sushicore.proc_memory import RecordingRunner
from typer.testing import CliRunner, Result

from sushiblas import cli, command_context
from sushiblas.command_context import CommandContext
from sushiblas.config import Config
from sushiblas.services.build_environment_memory import MemoryBuildEnvironment
from sushiblas.services.executable_lookup_memory import MemoryExecutableLookup
from sushiblas.services.package_source import PackageSource
from sushiblas.services.package_source_memory import MemoryPackageSource
from sushiblas.services.project import ProjectService

K_EXECUTABLE = "quickstart_demo"
#: The trees a plain build and a benchmark find, below the checkout.
K_DEBUG_TREE = Path("build") / "default-debug"
K_RELEASE_TREE = Path("build") / "default-release"


def make_config(**fields) -> Config:
    """Returns the real configuration of a Linux checkout that resolves no vcpkg tree and no package.

    Args:
        **fields: Fields of :class:`Config` to set beside the defaults of this module.
    """
    settings = {"cxx": "clang++", "generator": "Ninja", "platform": "linux", "use_vcpkg": False}
    settings.update(fields)
    return Config(**settings)


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


def make_service(root: Path, config: Config | None = None, packages: PackageSource | None = None,
                 environment: Mapping[str, str] | None = None,
                 ) -> tuple[RecordingRunner, RecordingConsole, ProjectService]:
    """Returns a recording runner and console and the real service, with its real CMake driver, on them.

    The service finds one executable, K_EXECUTABLE, directly under any tree. The packages are those
    of *packages*, none when it is not given, and every tree's subprocesses run under
    *environment*, an empty one when it is not given.
    """
    console = RecordingConsole()
    runner = RecordingRunner(console)
    settings = config or make_config()
    service = ProjectService(
        console=console,
        runner=runner,
        driver=CMakeDriver(console, runner),
        index=MemoryExecutableLookup((K_EXECUTABLE,)),
        build_env=MemoryBuildEnvironment(dict(environment or {})),
        find_root=lambda: root,
        load_config=lambda: settings,
        packages=packages or MemoryPackageSource(),
    )
    return runner, console, service


def make_commands(root: Path, config: Config | None = None,
                  packages: PackageSource | None = None) -> Commands:
    """Returns the commands over the checkout *root*, whose service uses *config* and *packages* when given."""
    runner, console, service = make_service(root, config, packages)
    context = CommandContext(
        service_factory=lambda: service, console=console, project_root=lambda: root)
    return Commands(root, runner, console, context)
