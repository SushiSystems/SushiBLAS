# check_tier_contents.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks what the foundation, kernel, engine.hpp, operation and recorder layers of SushiBLAS hold.

The tier order, upward includes and private reach are the shared check_layering.py's rules.

Usage: python tools/layering/check_tier_contents.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".glsl", ".comp", ".vert", ".frag"})

K_MODULES_FOLDER = "modules"
K_KERNEL_TIER = "kernels"
K_OPERATION_TIER = "operations"
K_INDEPENDENT_FOUNDATION_FOLDERS: tuple[str, ...] = tuple(
    f"modules/foundation/{module}/include"
    for module in ("arithmetic", "launch", "reduction", "reference", "substitution", "orthonormal",
                   "sparse_assembly", "geometry")
)
K_ENGINE_HEADER = "modules/engine/engine/include/SushiBLAS/engine/engine.hpp"
K_RECORDER_HEADER = "modules/engine/engine/include/SushiBLAS/engine/task_recorder.hpp"
K_GRAPH_PRIMITIVES_FOLDER = "modules/foundation/graph_primitives/"
K_UNSCREENED_OPERATION_MODULES = frozenset({"io"})
K_KERNEL_FORBIDDEN_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]'
    r'(SushiBLAS/(?:engine|operands|functor_map|tensor|sparse_storage)/[^>"]+)[>"]'
)
K_FOUNDATION_FORBIDDEN_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]'
    r'(SushiBLAS/(?:engine|operands|functor_map|tensor|sparse_storage|[a-z_]+_kernels)/[^>"]+)[>"]'
)
K_ENGINE_COMPOSED_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]'
    r'(SushiBLAS/(?:engine|operands|functor_map|io|experimental|blas|elementwise|nonlinear'
    r'|reductions|random|logic|sparse|spatial|signal|blend|index)/[^>"]+)[>"]'
)
K_ENGINE_BRICKS = frozenset({
    "SushiBLAS/engine/rng_stream.hpp",
    "SushiBLAS/engine/scratch_pool.hpp",
    "SushiBLAS/engine/execution_mode.hpp",
})
K_ENGINE_MEMBER_CALL = re.compile(r"\bengine_?\s*(?:\.|->)\s*([A-Za-z_]\w*)\s*\(")
K_OPERATION_ALLOWED_MEMBERS: frozenset[str] = frozenset()
K_RECORDING_VERBS: tuple[str, ...] = (
    "task",
    "kernel",
    "state_rule",
    "host",
    "untracked_kernel",
    "dynamic_kernel",
    "coupled_rule",
    "raw_kernel",
    "captured_kernel",
    "untracked_raw_kernel",
)
K_GRAPH_INSERTION = re.compile(r"(?:\.|->)\s*(" + "|".join(K_RECORDING_VERBS) + r")\s*\(")
K_LINE_COMMENT = re.compile(r"^\s*(?://|\*|/\*)")


@dataclass(frozen=True, slots=True)
class KnownViolation:
    """Holds one accepted layering violation: its rule, file, symbol and reason."""

    rule: str
    file: str
    symbol: str
    reason: str


K_KNOWN_VIOLATIONS: tuple[KnownViolation, ...] = ()


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds one repository root and its accepted violations."""

    root: Path
    known: tuple[KnownViolation, ...]


@dataclass(frozen=True, slots=True)
class Violation:
    """Holds one layering violation found in a file, before the known table screens it."""

    path: Path
    line: int
    symbol: str


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given repository folder with the known table."""
    return [Repository(path, K_KNOWN_VIOLATIONS) for path in paths if path.is_dir()]


def _source_lines(folder: Path) -> Iterator[tuple[Path, int, str]]:
    """Yields every line of every C-family source file below a folder."""
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in K_SOURCE_SUFFIXES:
            yield from _file_lines(path)


def _file_lines(path: Path) -> Iterator[tuple[Path, int, str]]:
    """Yields every line of one file with its number."""
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), start=1):
        yield path, number, line


def _include_matches(lines: Iterable[tuple[Path, int, str]], pattern: re.Pattern[str]) -> Iterator[Violation]:
    """Yields a violation for each line whose include matches a pattern."""
    for path, number, line in lines:
        match = pattern.match(line)
        if match:
            yield Violation(path, number, match.group(1))


def _module_folders(root: Path, tier: str | None = None) -> Iterator[Path]:
    """Yields each module folder below root/modules, of one tier or of every tier."""
    base = root / K_MODULES_FOLDER
    if not base.is_dir():
        return
    for tier_folder in sorted(base.iterdir()):
        if tier_folder.is_dir() and tier in (None, tier_folder.name):
            yield from sorted(folder for folder in tier_folder.iterdir() if folder.is_dir())


def _header_lines(root: Path, tier: str | None = None) -> Iterator[tuple[Path, int, str]]:
    """Yields the lines of every file under the include/ tree of each module."""
    for module in _module_folders(root, tier):
        yield from _source_lines(module / "include")


def _operation_source_lines(root: Path) -> Iterator[tuple[Path, int, str]]:
    """Yields the lines of the sources of each operation family, which IO is not."""
    for module in _module_folders(root, K_OPERATION_TIER):
        if module.name not in K_UNSCREENED_OPERATION_MODULES:
            yield from _source_lines(module / "source")


def _source_tree_lines(root: Path) -> Iterator[tuple[Path, int, str]]:
    """Yields the lines of every file under the source/ tree of each module."""
    for module in _module_folders(root):
        yield from _source_lines(module / "source")


def find_kernel_includes(root: Path) -> Iterator[Violation]:
    """Yields each include of an engine-tier header or a tensor header from a kernel."""
    yield from _include_matches(_header_lines(root, K_KERNEL_TIER), K_KERNEL_FORBIDDEN_INCLUDE)


def find_foundation_includes(root: Path) -> Iterator[Violation]:
    """Yields each engine-tier, kernel or tensor include made by a foundation module."""
    for folder in K_INDEPENDENT_FOUNDATION_FOLDERS:
        yield from _include_matches(_source_lines(root / folder), K_FOUNDATION_FORBIDDEN_INCLUDE)


def find_engine_header_includes(root: Path) -> Iterator[Violation]:
    """Yields each include from engine.hpp of a view, a seam or an experimental header."""
    for violation in _include_matches(_file_lines(root / K_ENGINE_HEADER), K_ENGINE_COMPOSED_INCLUDE):
        if violation.symbol not in K_ENGINE_BRICKS:
            yield violation


def _engine_calls(lines: Iterable[tuple[Path, int, str]], allowed: frozenset[str]) -> Iterator[Violation]:
    """Yields each call on an engine or engine_ variable to a member outside the allowed set."""
    for path, number, line in lines:
        if K_LINE_COMMENT.match(line):
            continue
        for member in K_ENGINE_MEMBER_CALL.findall(line):
            if member not in allowed:
                yield Violation(path, number, member)


def find_operation_engine_calls(root: Path) -> Iterator[Violation]:
    """Yields each call on an engine or engine_ variable from an operation source."""
    yield from _engine_calls(_operation_source_lines(root), K_OPERATION_ALLOWED_MEMBERS)


def _may_record(file: str) -> bool:
    """Returns whether a repository-relative file is TaskRecorder or a graph primitive."""
    return file == K_RECORDER_HEADER or file.startswith(K_GRAPH_PRIMITIVES_FOLDER)


def find_graph_insertions(root: Path) -> Iterator[Violation]:
    """Yields each call of a recording member of SushiRuntime::Recorder outside TaskRecorder and the graph primitives."""
    for path, number, line in chain(_header_lines(root), _source_tree_lines(root)):
        if K_LINE_COMMENT.match(line) or _may_record(_relative(path, root)):
            continue
        for call in K_GRAPH_INSERTION.findall(line):
            yield Violation(path, number, call)


@cache
def _relative(path: Path, root: Path) -> str:
    """Returns a path relative to the repository root, with forward slashes."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _known_for(repository: Repository, rule: str) -> dict[tuple[str, str], KnownViolation]:
    """Returns the known entries of one rule, keyed by (file, symbol)."""
    return {(entry.file, entry.symbol): entry for entry in repository.known if entry.rule == rule}


def _screened_rule(rule: str, find: Callable[[Path], Iterator[Violation]]) -> Callable[[Repository], Iterator[Issue]]:
    """Returns a rule that reports violations missing from the known table and stale entries."""
    def apply(repository: Repository) -> Iterator[Issue]:
        known = _known_for(repository, rule)
        seen: set[tuple[str, str]] = set()
        for violation in find(repository.root):
            key = (_relative(violation.path, repository.root), violation.symbol)
            seen.add(key)
            if key not in known:
                yield Issue(violation.path, violation.line, f"layering violation: {violation.symbol}")
        for key in sorted(set(known) - seen):
            yield Issue(Path(__file__), 1, f"stale K_KNOWN_VIOLATIONS entry: {key[0]} {key[1]}")
    apply.__doc__ = find.__doc__
    return apply


def known_notes(repository: Repository, rules: Iterable[str]) -> Iterator[str]:
    """Yields one note line for each known entry of the given rules that still occurs."""
    for rule in rules:
        find = K_SCREENED_FINDERS.get(rule)
        if find is None:
            continue
        known = _known_for(repository, rule)
        reported: set[tuple[str, str]] = set()
        for violation in find(repository.root):
            key = (_relative(violation.path, repository.root), violation.symbol)
            if key in known and key not in reported:
                reported.add(key)
                yield f"note: {key[0]}:{violation.line}: {rule}: known: {key[1]} ({known[key].reason})"


K_SCREENED_FINDERS: dict[str, Callable[[Path], Iterator[Violation]]] = {
    "rule_foundation_independence": find_foundation_includes,
    "rule_kernel_includes": find_kernel_includes,
    "rule_engine_header_includes": find_engine_header_includes,
    "rule_operation_engine_calls": find_operation_engine_calls,
    "rule_graph_insertions": find_graph_insertions,
}

rule_foundation_independence = _screened_rule("rule_foundation_independence", find_foundation_includes)
rule_kernel_includes = _screened_rule("rule_kernel_includes", find_kernel_includes)
rule_engine_header_includes = _screened_rule("rule_engine_header_includes", find_engine_header_includes)
rule_operation_engine_calls = _screened_rule("rule_operation_engine_calls", find_operation_engine_calls)
rule_graph_insertions = _screened_rule("rule_graph_insertions", find_graph_insertions)

K_CHECKER = Checker(
    description="Checks what the foundation, kernel, engine.hpp, operation and recorder layers hold.",
    collect=collect,
    rules={
        "rule_foundation_independence": rule_foundation_independence,
        "rule_kernel_includes": rule_kernel_includes,
        "rule_engine_header_includes": rule_engine_header_includes,
        "rule_operation_engine_calls": rule_operation_engine_calls,
        "rule_graph_insertions": rule_graph_insertions,
    },
)


def _print_notes(argv: list[str]) -> None:
    """Prints the known-violation notes for the paths and rule a command line names."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rule")
    arguments, _ = parser.parse_known_args(argv)
    if arguments.report:
        return
    rules = [arguments.rule] if arguments.rule else list(K_SCREENED_FINDERS)
    for repository in collect(arguments.paths or [Path.cwd()]):
        for note in known_notes(repository, rules):
            print(note)


if __name__ == "__main__":
    _print_notes(sys.argv[1:])
    sys.exit(run(K_CHECKER))
