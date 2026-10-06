# cli.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Declares the `sb` commands; each parses its arguments and calls ``sushiblas.services``."""

from __future__ import annotations

from typing import Optional

import typer
from sushicore import entry
from sushicore.diag_commands import register_diagnostic_commands
from sushicore.provision.commands import ModuleProvision, register_provision_commands
from sushicore.root_options import register_root_options
from sushicore.typer_help import help_group

from . import console
from .aliases import K_ALIASES
from .config import PROFILE, find_project_root, load_config
from .services import project as project_svc
from .services.diag import K_DIAGNOSTICS as K_DIAGNOSTICS_SERVICE
from .services.project import BuildType, Suite

K_PROJECT = "Project"
K_DOCUMENTATION = "Documentation"
K_DIAGNOSTICS = "Diagnostics"
K_DISTRIBUTION = "sushiblas-cli"

_help_group = help_group(console.current)

app = typer.Typer(
    name="sb",
    cls=_help_group,
    help="SushiBLAS developer CLI — build, test, and run SushiBLAS.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

register_root_options(app, distribution=K_DISTRIBUTION)


@app.command(
    "build",
    rich_help_panel=K_PROJECT,
    epilog="sb build\n"
           "sb build --type debug  # Build with debug settings\n"
           "sb build --clean  # Start from an empty build tree\n"
           "sb build -D SB_LOG_LEVEL=INFO  # Pass a CMake cache variable through",
)
def build(
    type: BuildType = typer.Option(
        BuildType.release, "--type", "-t", case_sensitive=False,
        help="Build type: release | debug | relwithdebinfo. "
             "Changing it empties the build tree first."),
    clean: bool = typer.Option(
        False, "--clean", help="Delete the build tree before configuring."),
    define: Optional[list[str]] = typer.Option(
        None, "--define", "-D", metavar="VAR=VALUE",
        help="Pass a CMake cache variable through to configure. Repeatable. "
             "Quote values containing ';', e.g. "
             '-D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86".'),
) -> None:
    """Configure and build the project.

    Cache variables set with -D persist in the build tree, so dropping the
    flag on a later run does not unset them; pass VAR= to clear one, or
    --clean to start over. A different --type also starts over: the tree
    is removed, and the variables with it.
    """
    raise typer.Exit(project_svc.build(type, clean, define))


@app.command(
    "test",
    rich_help_panel=K_PROJECT,
    epilog="sb test\n"
           "sb test --suite unit\n"
           "sb test --filter 'Gemm.*'  # Match Suite.Case test names\n"
           "sb test --repeat 5  # Re-run each test until one fails",
)
def test(
    suite: Suite = typer.Option(
        Suite.functional, "--suite", "-s", case_sensitive=False,
        help="Which CTest label group to run ('all' runs every CTest test; "
             "'package' instead builds tests/package/ against an install "
             "prefix and is not included in 'all')."),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="ctest -R regex over 'Suite.Case' test names."),
    repeat: int = typer.Option(
        0, "--repeat", "-r", min=0,
        help="Re-run each test up to N times, stopping on the first failure "
             "(ctest --repeat until-fail). Handy for hunting flaky tests."),
) -> None:
    """Run the test suite via CTest labels.

    For GTest-level knobs (per-case filtering, shuffle, break-on-failure) run
    the binary directly: sb run sushiblas_tests -- --gtest_filter=Gemm.*

    --suite package is the exception: it installs the package and builds the
    out-of-tree consumer in tests/package/ against it, which is the only check
    of what find_package(SushiBLAS) hands a third party. It runs no CTest, so
    --filter and --repeat do not apply to it.
    """
    raise typer.Exit(project_svc.test(suite, filter, repeat))


@app.command(
    "run",
    rich_help_panel=K_PROJECT,
    epilog="sb run sushiblas_example\n"
           "sb run --sort  # Pick an executable from a list\n"
           "sb run sushiblas_tests -- --gtest_filter='Unit_GEMMTest.*'  # Flags after -- go to it",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(
        None, help="Executable name to run (exact, then substring match)."),
    sort: bool = typer.Option(
        False, "--sort", help="Interactively pick an executable."),
) -> None:
    """Run a built executable. Args after `--` are forwarded to it."""
    raise typer.Exit(project_svc.run(target=target, sort=sort, app_args=list(ctx.args)))


@app.command(
    "clean",
    rich_help_panel=K_PROJECT,
    epilog="sb clean",
)
def clean() -> None:
    """Remove the build/ tree."""
    raise typer.Exit(project_svc.clean())


@app.command(
    "docs",
    rich_help_panel=K_DOCUMENTATION,
    epilog="sb docs",
)
def docs() -> None:
    """Generate the Doxygen API reference."""
    raise typer.Exit(project_svc.docs())


K_ALIASES.command(app, "doxygen", docs, "sb doxygen")

register_diagnostic_commands(
    app, K_DIAGNOSTICS_SERVICE, program=PROFILE.program, panel=K_DIAGNOSTICS)

register_provision_commands(app, ModuleProvision(
    profile=PROFILE,
    project_root=find_project_root,
    load_config=load_config,
    console=lambda: console,
    locate=lambda name: load_config().locate_sibling(find_project_root(), name),
    panel=K_DIAGNOSTICS,
    uses_base=True,
))


def _report_error(message: str) -> None:
    """Prints one error line through the console, which is built only now."""
    console.error(message)


def main() -> None:
    """Runs `sb`; a failure the user can act on leaves one line and exit code 1."""
    entry.run(app, _report_error)


if __name__ == "__main__":
    main()
