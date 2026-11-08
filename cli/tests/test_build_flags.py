# test_build_flags.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for what a build decides from its options alone: its -D flags, its tree and its packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from sushiblas.services.build_type import BuildType
from sushiblas.services.errors import InvalidDefineError, LeanBenchmarksError
from sushiblas.services.library_packages import package_options, package_type
from sushiblas.services.project import (
    benchmarks_flags,
    caller_flags,
    distributed_flags,
    lean_flags,
    refuse_lean_benchmarks,
    validate_defines,
)
from sushiblas.services.sanitizer import Sanitizer, sanitize_flags
from sushiblas.services.tree_choice import TreeChoice

pytestmark = pytest.mark.small

#: A checkout path that is only a name; no test reads or writes it.
K_ROOT = Path("/checkout")
K_INSTRUMENTED_LOG = ("--sanitize", "thread")


@pytest.mark.parametrize(
    ("defines", "expected"),
    [
        (["SB_ENABLE_LOGGING=OFF", "SB_NATIVE_ARCH="], ["-DSB_ENABLE_LOGGING=OFF", "-DSB_NATIVE_ARCH="]),
        ([], []),
        (None, []),
    ],
)
def test_validate_defines_turns_each_entry_into_one_flag_in_order(defines, expected):
    """Each VAR=VALUE entry becomes -DVAR=VALUE, in the order given; an empty value is kept."""
    assert validate_defines(defines) == expected


def test_validate_defines_refuses_an_entry_without_a_value():
    """An entry with no '=' raises and names the entry."""
    with pytest.raises(InvalidDefineError) as raised:
        validate_defines(["SB_ENABLE_LOGGING=OFF", "NOVALUE"])
    assert "NOVALUE" in str(raised.value)


@pytest.mark.parametrize(
    ("benchmarks", "expected"),
    [(True, ["-DSB_BUILD_BENCHMARKS=ON"]), (False, ["-DSB_BUILD_BENCHMARKS=OFF"]), (None, [])],
)
def test_benchmarks_flags_switch_the_benchmark_programs(benchmarks, expected):
    """--benchmarks and --no-benchmarks become one SB_BUILD_BENCHMARKS flag; none leaves it out."""
    assert benchmarks_flags(benchmarks) == expected


@pytest.mark.parametrize(
    ("distributed", "expected"),
    [(True, ["-DSB_ENABLE_DISTRIBUTED=ON"]), (False, ["-DSB_ENABLE_DISTRIBUTED=OFF"]), (None, [])],
)
def test_distributed_flags_switch_the_cluster_module(distributed, expected):
    """--distributed and --no-distributed become one SB_ENABLE_DISTRIBUTED flag; none leaves it out."""
    assert distributed_flags(distributed) == expected


@pytest.mark.parametrize(
    ("sanitizer", "expected"),
    [
        (Sanitizer.ADDRESS_UNDEFINED, ["-DSB_SANITIZE=address,undefined"]),
        (Sanitizer.THREAD, ["-DSB_SANITIZE=thread"]),
        (Sanitizer.NONE, ["-DSB_SANITIZE="]),
        (None, ["-DSB_SANITIZE="]),
    ],
)
def test_sanitize_flags_set_the_sanitizer_set(sanitizer, expected):
    """--sanitize becomes one SB_SANITIZE flag; the plain tree, left out or none, clears it."""
    assert sanitize_flags(sanitizer) == expected


@pytest.mark.parametrize(
    ("lean", "expected"),
    [
        (True, ["-DSB_BUILD_TESTS=OFF", "-DSB_BUILD_EXAMPLES=OFF", "-DSB_BUILD_BENCHMARKS=OFF"]),
        (False, ["-DSB_BUILD_TESTS=ON", "-DSB_BUILD_EXAMPLES=ON"]),
    ],
)
def test_lean_flags_leave_out_the_tests_the_examples_and_the_benchmarks(lean, expected):
    """A lean build turns the tests, the examples and the benchmarks off; a plain one asks for the first two."""
    assert lean_flags(lean) == expected


def test_caller_flags_put_a_define_after_the_switch_it_outranks():
    """A -D for the same variable as a switch comes after it, so the -D wins."""
    assert caller_flags(True, None, None, ["SB_BUILD_BENCHMARKS=OFF"]) == [
        "-DSB_BUILD_BENCHMARKS=ON", "-DSB_SANITIZE=", "-DSB_BUILD_BENCHMARKS=OFF"]


def test_caller_flags_order_the_switches_before_the_defines():
    """The benchmark, distributed and sanitizer switches come in that order, then the caller's -D entries."""
    assert caller_flags(False, True, Sanitizer.THREAD, ["A=1"]) == [
        "-DSB_BUILD_BENCHMARKS=OFF", "-DSB_ENABLE_DISTRIBUTED=ON", "-DSB_SANITIZE=thread", "-DA=1"]


def test_refuse_lean_benchmarks_refuses_a_lean_build_that_asks_for_the_benchmarks():
    """--lean with --benchmarks asks for opposite things and raises."""
    with pytest.raises(LeanBenchmarksError):
        refuse_lean_benchmarks(True, True)


@pytest.mark.parametrize(("lean", "benchmarks"), [(True, None), (True, False), (False, True)])
def test_refuse_lean_benchmarks_accepts_every_other_pair(lean, benchmarks):
    """A lean build that leaves the benchmarks alone or off, and a plain one with them, pass."""
    assert refuse_lean_benchmarks(lean, benchmarks) is None


@pytest.mark.parametrize(
    ("tree", "requested", "expected"),
    [
        (TreeChoice(), None, Path("build") / "default-debug"),
        (TreeChoice(BuildType.RELEASE), None, Path("build") / "default-release"),
        (TreeChoice(BuildType.RELWITHDEBINFO), None, Path("build") / "default-relwithdebinfo"),
        (TreeChoice(BuildType.DEBUG, Sanitizer.ADDRESS_UNDEFINED), None, Path("build") / "asan-debug"),
        (TreeChoice(BuildType.RELEASE, Sanitizer.THREAD), None, Path("build") / "tsan-release"),
        (TreeChoice(BuildType.DEBUG, Sanitizer.NONE), None, Path("build") / "default-debug"),
        (TreeChoice(BuildType.RELEASE, Sanitizer.THREAD), "build/pkg-x", Path("build") / "pkg-x"),
    ],
)
def test_tree_choice_directory_follows_the_type_the_sanitizer_and_the_override(
    tree, requested, expected
):
    """Each type and sanitizer set has its own tree below build/; --build-dir names it whatever they are."""
    assert tree.directory(K_ROOT, requested) == K_ROOT / expected


@pytest.mark.parametrize(
    ("build_type", "sanitizer", "expected"),
    [
        ("debug", None, "release"),
        ("release", None, "release"),
        ("relwithdebinfo", None, "relwithdebinfo"),
        ("relwithdebinfo", Sanitizer.NONE, "relwithdebinfo"),
        ("relwithdebinfo", Sanitizer.ADDRESS_UNDEFINED, "release"),
        ("debug", Sanitizer.ADDRESS_UNDEFINED, "release"),
        ("relwithdebinfo", Sanitizer.THREAD, "relwithdebinfo"),
        ("release", Sanitizer.THREAD, "relwithdebinfo"),
    ],
)
def test_package_type_names_the_closure_a_tree_links(build_type, sanitizer, expected):
    """Debug, Release and an ASan tree link Release packages; RelWithDebInfo and a TSan tree link RelWithDebInfo."""
    assert package_type(build_type, sanitizer) == expected


@pytest.mark.parametrize(
    ("distributed", "sanitizer", "expected"),
    [
        (False, None, {}),
        (False, Sanitizer.ADDRESS_UNDEFINED, {}),
        (True, None, {"sushiruntime": ("--distributed",)}),
        (False, Sanitizer.THREAD, {
            "sushilog": K_INSTRUMENTED_LOG,
            "sushiprofiler": K_INSTRUMENTED_LOG,
            "sushiruntime": ("--type", "tsan"),
        }),
        (True, Sanitizer.THREAD, {
            "sushilog": K_INSTRUMENTED_LOG,
            "sushiprofiler": K_INSTRUMENTED_LOG,
            "sushiruntime": ("--distributed", "--type", "tsan"),
        }),
    ],
)
def test_package_options_ask_for_the_distributed_runtime_and_instrumented_packages(
    distributed, sanitizer, expected
):
    """--distributed reaches the runtime package; a TSan tree instruments all three, keeping --distributed."""
    assert package_options(distributed, sanitizer) == expected
