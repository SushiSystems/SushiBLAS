"""SushiBLAS developer CLI (`sb`).

Thin Typer layer: commands only parse arguments and delegate to the service
layer in ``sushiblas.services``.
"""

from __future__ import annotations

from typing import Optional

import typer
from sushicore.provision.commands import ModuleProvision, register_provision_commands
from sushicore.typer_help import help_group

from . import console
from .config import PROFILE, find_project_root, load_config
from .services import diag as diag_svc
from .services import project as project_svc
from .services.project import BuildType, Suite

K_PROJECT = "Project"
K_DOCUMENTATION = "Documentation"
K_DIAGNOSTICS = "Diagnostics"

_help_group = help_group(console.current)

app = typer.Typer(
    name="sb",
    cls=_help_group,
    help="SushiBLAS developer CLI — build, test, and run SushiBLAS.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


# --------------------------------------------------------------------------- #
# build / test / run / clean / doxygen
# --------------------------------------------------------------------------- #
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
        help="Build type: release | debug | relwithdebinfo."),
    clean: bool = typer.Option(
        False, "--clean", help="Delete the build tree before configuring."),
    define: Optional[list[str]] = typer.Option(
        None, "--define", "-D", metavar="VAR=VALUE",
        help="Pass a CMake cache variable through to configure. Repeatable. "
             "Quote values containing ';', e.g. "
             '-D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86".'),
):
    """Configure and build the project.

    Cache variables set with -D persist in the build tree, so dropping the
    flag on a later run does not unset them; pass VAR= to clear one, or
    --clean to start over.
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
):
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
):
    """Run a built executable. Args after `--` are forwarded to it."""
    raise typer.Exit(project_svc.run(target=target, sort=sort, app_args=list(ctx.args)))


@app.command(
    "clean",
    rich_help_panel=K_PROJECT,
    epilog="sb clean",
)
def clean():
    """Remove the build/ tree."""
    raise typer.Exit(project_svc.clean())


@app.command(
    "doxygen",
    rich_help_panel=K_DOCUMENTATION,
    epilog="sb doxygen",
)
def doxygen():
    """Generate Doxygen documentation."""
    raise typer.Exit(project_svc.doxygen())


# --------------------------------------------------------------------------- #
# config / env (diagnostics)
# --------------------------------------------------------------------------- #
@app.command(
    "config",
    rich_help_panel=K_DIAGNOSTICS,
    epilog="sb config",
)
def config():
    """Print the resolved config and where each value came from."""
    raise typer.Exit(diag_svc.config_show())


@app.command(
    "env",
    rich_help_panel=K_DIAGNOSTICS,
    epilog="sb env\n"
           "sb env --all  # Show every variable, not just build-relevant ones",
)
def env(
    all: bool = typer.Option(
        False, "--all", help="Show every variable, not just build-relevant ones."),
):
    """Print the environment cmake/ctest/run subprocesses execute under."""
    raise typer.Exit(diag_svc.env_dump(show_all=all))


register_provision_commands(app, ModuleProvision(
    profile=PROFILE,
    project_root=find_project_root,
    load_config=load_config,
    console=lambda: console,
    locate=lambda name: load_config().locate_sibling(find_project_root(), name),
    panel=K_DIAGNOSTICS,
    uses_base=True,
))


if __name__ == "__main__":
    app()
