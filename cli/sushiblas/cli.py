# cli.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Declares the `sb` commands; each parses its arguments and calls ``sushiblas.services``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, Optional

import typer
from sushicore import entry
from sushicore.diag_commands import register_diagnostic_commands
from sushicore.docs_bundle import ApiSource, register_docs_commands
from sushicore.provision.commands import ModuleProvision, register_provision_commands
from sushicore.provision.package_command import register_package_command
from sushicore.root_options import register_root_options
from sushicore.typer_help import help_group

from . import console
from .aliases import K_ALIASES
from .config import PROFILE, find_project_root, load_config
from .services.build_type import K_BENCHMARK_BUILD_TYPE, K_DEFAULT_BUILD_TYPE, BuildType
from .services.diag import K_DIAGNOSTICS as K_DIAGNOSTICS_SERVICE
from .services.errors import CLIOptionError, PackageOptionError, ProjectError
from .services.project import ProjectService
from .services.project_factory import create_project_service
from .services.sanitizer import Sanitizer
from .services.test_label import TestLabel
from .services.tree_choice import TreeChoice

K_PROJECT = "Project"
K_DOCUMENTATION = "Documentation"
K_DIAGNOSTICS = "Diagnostics"
K_DISTRIBUTION = "sushiblas-cli"
K_API_XML = Path("build") / "docs" / "api-site" / "xml"

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


def _type_option(default: BuildType) -> BuildType:
    """Returns the --type option of a command that finds a tree, with *default* as its type."""
    return typer.Option(
        default, "--type", "-t", case_sensitive=False,
        help="Build type of the tree: release | debug | relwithdebinfo. "
             "Each type has a tree of its own.")


def _sanitize_option() -> Optional[Sanitizer]:
    """Returns the --sanitize option of a command that finds a tree."""
    return typer.Option(
        None, "--sanitize", case_sensitive=False,
        help="Sanitizer tree: none | address-undefined | thread. Left out, the plain tree.")


def _service() -> ProjectService:
    """Returns a ProjectService that reports through this run's console."""
    return create_project_service(console.current())


def _code_of(step: Callable[[], int]) -> int:
    """Returns the code *step* returns, or prints a ProjectError and returns its code."""
    try:
        return step()
    except ProjectError as error:
        console.error(str(error))
        return error.exit_code


def _exit_with(step: Callable[[], int]) -> NoReturn:
    """Runs *step* and exits with its code, or prints a ProjectError and exits with its code."""
    raise typer.Exit(_code_of(step))


@app.command(
    "build",
    rich_help_panel=K_PROJECT,
    epilog="sb build\n"
           "sb build --type release  # The Release tree, build/default-release\n"
           "sb build --clean  # Start from an empty build tree\n"
           "sb build --benchmarks  # Also build the benchmark programs\n"
           "sb build --distributed  # Also build the cluster module\n"
           "sb build --jobs 4  # Run at most 4 compile jobs at once\n"
           "sb build --report  # Print what each kind of build edge cost\n"
           "sb build --sanitize address-undefined  # The ASan and UBSan tree, build/asan-debug\n"
           "sb build -D SB_ENABLE_LOGGING=OFF  # Pass a CMake cache variable through\n"
           "sb build --build-dir build/pkg-x --lean  # What a consumer's package build runs",
)
def build(
    type: BuildType = typer.Option(
        K_DEFAULT_BUILD_TYPE, "--type", "-t", case_sensitive=False,
        help="Build type: release | debug | relwithdebinfo. "
             "Each type builds in a tree of its own, build/<name>-<type>."),
    clean: bool = typer.Option(
        False, "--clean", help="Delete the build tree before configuring."),
    benchmarks: Optional[bool] = typer.Option(
        None, "--benchmarks/--no-benchmarks",
        help="Build the benchmark programs under tests/, or stop building them. "
             "Left out, the setting of the last build stays."),
    distributed: Optional[bool] = typer.Option(
        None, "--distributed/--no-distributed",
        help="Build the cluster module over SushiRuntime's distributed layer, or stop "
             "building it. Left out, the setting of the last build stays."),
    sanitize: Optional[Sanitizer] = typer.Option(
        None, "--sanitize", case_sensitive=False,
        help="Sanitizers for host code: none | address-undefined | thread. "
             "Each set builds in a tree of its own: build/asan-<type> or build/tsan-<type>."),
    define: Optional[list[str]] = typer.Option(
        None, "--define", "-D", metavar="VAR=VALUE",
        help="Pass a CMake cache variable through to configure. Repeatable. "
             "Quote values containing ';', e.g. "
             '-D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86".'),
    jobs: Optional[int] = typer.Option(
        None, "--jobs", "-j", min=1,
        help="Compile jobs run at once. When omitted it is one per hardware thread, and "
             "CMAKE_BUILD_PARALLEL_LEVEL is honoured."),
    build_dir: Optional[str] = typer.Option(
        None, "--build-dir", metavar="PATH",
        help="Build in this tree, below the checkout, instead of build/<name>-<type>."),
    lean: bool = typer.Option(
        False, "--lean",
        help="Build no tests, examples or benchmarks; what a consumer's package build of "
             "SushiBLAS uses."),
    report: bool = typer.Option(
        False, "--report",
        help="After the build, print what each kind of build edge cost in time and memory "
             "and write build_report.json into the tree."),
) -> None:
    """Configure and build the project.

    Cache variables set with -D persist in the build tree, so dropping the
    flag on a later run does not unset them; pass VAR= to clear one, or
    --clean to start over. A different --type builds in another tree, and a
    tree configured as one type is never configured as another.
    """
    console.header("Project Build")

    def step() -> int:
        """Builds and announces success."""
        rc = _service().build(type, clean, define, benchmarks, sanitize, jobs, distributed,
                              lean, build_dir, report)
        console.success("Build completed successfully!")
        return rc

    _exit_with(step)


@app.command(
    "test",
    rich_help_panel=K_PROJECT,
    epilog="sb test  # The small and medium tests\n"
           "sb test --label small\n"
           "sb test --label large  # Needs a real GPU\n"
           "sb test --label quarantined\n"
           "sb test --filter 'Gemm.*'  # Match Suite.Case test names\n"
           "sb test --repeat 5  # Re-run each test until one fails\n"
           "sb test --cli  # The pytest tests of this CLI\n"
           "sb test --package  # The out-of-tree consumer of the installed package",
)
def test(
    label: Optional[list[TestLabel]] = typer.Option(
        None, "--label", "-l", case_sensitive=False,
        help="Run the tests of this label: small | medium | large | quarantined. Repeatable. "
             "Left out, the small and medium tests run. A quarantined test runs only when "
             "quarantined is named."),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="ctest -R regex over 'Suite.Case' test names."),
    repeat: int = typer.Option(
        0, "--repeat", "-r", min=0,
        help="Re-run each test up to N times, stopping on the first failure "
             "(ctest --repeat until-fail). Handy for hunting flaky tests."),
    jobs: Optional[int] = typer.Option(
        None, "--jobs", "-j", min=1,
        help="Tests run at once. When omitted ctest decides, and "
             "CTEST_PARALLEL_LEVEL is honoured."),
    cli: bool = typer.Option(
        False, "--cli", help="Run the pytest tests of this CLI instead of the CTest tests."),
    package: bool = typer.Option(
        False, "--package",
        help="Install the package and build tests/package/ against the install prefix "
             "instead of running the CTest tests."),
    type: BuildType = _type_option(K_DEFAULT_BUILD_TYPE),
    sanitize: Optional[Sanitizer] = _sanitize_option(),
) -> None:
    """Run the tests of the labels given.

    For GTest-level knobs (per-case filtering, shuffle, break-on-failure) run
    the binary directly: sb run sushiblas_blas_tests -- --gtest_filter=Gemm.*

    --package is the only check of what find_package(SushiBLAS) hands a third
    party. It runs no CTest, so --label, --filter, --repeat, --jobs and --cli do
    not apply to it. --cli runs pytest, so --filter, --repeat and --jobs do not
    apply to it.
    """
    console.header("Project Test")

    def step() -> int:
        """Runs the tests; a package run that returns has passed."""
        if package:
            if label or filter or repeat or jobs or cli:
                raise PackageOptionError()
            rc = _service().test_package(TreeChoice(type, sanitize))
            console.success("Package consumer built and ran against the install prefix.")
            return rc
        if cli:
            if filter or repeat or jobs:
                raise CLIOptionError()
            return _service().test_cli(label)
        return _service().test(label, filter, repeat, jobs, TreeChoice(type, sanitize))

    _exit_with(step)


@app.command(
    "bench",
    rich_help_panel=K_PROJECT,
    epilog="sb build --benchmarks\n"
           "sb bench  # Every benchmark program, in name order\n"
           "sb bench device_ceilings -- --out build/ceilings  # One scenario benchmark; its options go after --",
    context_settings={"allow_extra_args": True},
)
def bench(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(
        None, help="Scenario benchmark to run (exact, then substring match). "
                   "Every micro benchmark program runs when omitted."),
    type: BuildType = _type_option(K_BENCHMARK_BUILD_TYPE),
    sanitize: Optional[Sanitizer] = _sanitize_option(),
) -> None:
    """Run the built benchmark programs, or one scenario benchmark with its options.

    A benchmark measures and does not pass or fail, so it is not a test. Build
    the programs with sb build --benchmarks first. Args after `--` are forwarded
    to the scenario benchmark named.
    """
    if name is None and ctx.args:
        raise typer.BadParameter(
            "options for a benchmark go after its name and `--`.", param_hint="NAME")
    if name is not None and name.startswith("-"):
        raise typer.BadParameter(
            f"{name!r} is not a benchmark; options for the program go after `--`.",
            param_hint="NAME",
        )
    console.header("Project Bench")
    _exit_with(lambda: _service().bench(name, list(ctx.args), TreeChoice(type, sanitize)))


@app.command(
    "run",
    rich_help_panel=K_PROJECT,
    epilog="sb run quickstart_demo\n"
           "sb run --sort  # Pick an executable from a list\n"
           "sb run sushiblas_blas_tests -- --gtest_filter='Medium_GEMMTest.*'  # Flags after -- go to it",
    context_settings={"allow_extra_args": True},
)
def run(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(
        None, help="Executable name to run (exact, then substring match)."),
    sort: bool = typer.Option(
        False, "--sort", help="Interactively pick an executable."),
    type: BuildType = _type_option(K_DEFAULT_BUILD_TYPE),
    sanitize: Optional[Sanitizer] = _sanitize_option(),
) -> None:
    """Run a built executable. Args after `--` are forwarded to it."""
    if target is not None and target.startswith("-"):
        raise typer.BadParameter(
            f"{target!r} is not a target; options for the program go after `--`.",
            param_hint="TARGET",
        )
    console.header("Project Run")
    _exit_with(lambda: _service().run(
        target=target, sort=sort, app_args=list(ctx.args), tree=TreeChoice(type, sanitize)))


@app.command(
    "clean",
    rich_help_panel=K_PROJECT,
    epilog="sb clean\n"
           "sb clean --type release  # The Release tree\n"
           "sb clean --sanitize thread",
)
def clean(
    type: BuildType = _type_option(K_DEFAULT_BUILD_TYPE),
    sanitize: Optional[Sanitizer] = _sanitize_option(),
) -> None:
    """Remove one build tree: the Debug one unless --type says another."""
    console.header("Project Clean")

    def step() -> int:
        """Removes the tree and says what happened."""
        result = _service().clean(TreeChoice(type, sanitize))
        if result.removed:
            console.success(f"{result.tree} removed.")
        else:
            console.info(f"{result.tree} does not exist, nothing to clean.")
        return 0

    _exit_with(step)


def _build_api_reference() -> int:
    """Prints the header, generates the Doxygen API reference and returns the exit code."""
    console.header("Doxygen Generation")
    return _code_of(lambda: _service().docs())


def _api_source() -> ApiSource:
    """Returns how `sb` builds the API reference and where Doxygen leaves its XML."""
    return ApiSource(build=_build_api_reference, xml_dir=find_project_root() / K_API_XML)


def _docs_alias() -> None:
    """Builds the API reference; the callback behind the old spelling `sb doxygen`."""
    raise typer.Exit(_build_api_reference())


register_docs_commands(
    app,
    program=PROFILE.program,
    panel=K_DOCUMENTATION,
    project_root=find_project_root,
    api=_api_source,
    report=lambda line: console.success(line),
    group_cls=_help_group,
)

K_ALIASES.command(app, "doxygen", _docs_alias, "sb doxygen")

register_diagnostic_commands(
    app, K_DIAGNOSTICS_SERVICE, program=PROFILE.program, panel=K_DIAGNOSTICS)

_PROVISION = ModuleProvision(
    profile=PROFILE,
    project_root=find_project_root,
    load_config=load_config,
    console=lambda: console,
    locate=lambda name: load_config().locate_sibling(find_project_root(), name),
    panel=K_DIAGNOSTICS,
    uses_base=True,
)

register_provision_commands(app, _PROVISION)
register_package_command(app, replace(_PROVISION, panel=K_PROJECT))


def _report_error(message: str) -> None:
    """Prints one error line through the console, which is built only now."""
    console.error(message)


def main() -> None:
    """Runs `sb`; a failure the user can act on leaves one line and exit code 1."""
    entry.run(app, _report_error)


if __name__ == "__main__":
    main()
