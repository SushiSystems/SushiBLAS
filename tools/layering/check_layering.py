# check_layering.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks module tiers and the support, kernel, Engine, view and recorder layers of SushiBLAS.

Usage: python tools/layering/check_layering.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Checker, Issue, run  # noqa: E402

K_TIER_ORDER: tuple[str, ...] = ()
K_SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".cpp", ".glsl", ".comp", ".vert", ".frag"})
K_ANGLE_INCLUDE = re.compile(r'^\s*#\s*include\s*<[^/>]+/([^/>]+)/[^>]+>')
K_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
K_CMAKE_SOURCE_REACH = re.compile(r"(?:^|[/\s\"'])([A-Za-z0-9_]+)/source\b")

K_KERNEL_FOLDERS = ("kernels", "include/SushiBLAS/kernels")
K_SUPPORT_FOLDER = "include/SushiBLAS/support"
K_INCLUDE_FOLDER = "include/SushiBLAS"
K_ENGINE_HEADER = "include/SushiBLAS/engine.hpp"
K_UMBRELLA_HEADER = "include/SushiBLAS/SushiBLAS.h"
K_RECORDER_HEADER = "include/SushiBLAS/engine/task_recorder.hpp"
K_SOURCE_FOLDER = "src"
K_OPERATION_FOLDER = "src/engine"
K_KERNEL_FORBIDDEN_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]'
    r'(SushiBLAS/(?:engine\.hpp|engine/[^>"]+|tensor\.hpp'
    r'|sparse_(?:vector|matrix|builder)\.hpp))[>"]'
)
K_SUPPORT_FORBIDDEN_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]'
    r'(SushiBLAS/(?:engine\.hpp|engine/[^>"]+|kernels/[^>"]+|tensor\.hpp'
    r'|sparse_(?:vector|matrix|builder)\.hpp))[>"]'
)
K_ENGINE_COMPOSED_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"](SushiBLAS/(?:io\.hpp|experimental/[^>"]+|engine/[^>"]+))[>"]'
)
K_ENGINE_BRICKS = frozenset({
    "SushiBLAS/engine/rng_stream.hpp",
    "SushiBLAS/engine/scratch_pool.hpp",
    "SushiBLAS/engine/task_recorder.hpp",
    "SushiBLAS/engine/execution_mode.hpp",
})
K_ENGINE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](SushiBLAS/engine\.hpp)[>"]')
K_UMBRELLA_INCLUDE = re.compile(r"^\s*#\s*include\s*<SushiBLAS/([^>]+)>")
K_ENGINE_MEMBER_CALL = re.compile(r"\bengine_?\s*(?:\.|->)\s*([A-Za-z_]\w*)\s*\(")
K_OPERATION_ALLOWED_MEMBERS: frozenset[str] = frozenset()
K_VIEW_ALLOWED_MEMBERS = frozenset({"get_recorder", "get_rng"})
K_GRAPH_INSERTION = re.compile(r"\b(add_(?:task|node|native|offload_dynamic))\s*\(")
K_TASK_BODY_FOLDERS = ("kernels", "include/SushiBLAS", "src/engine")
K_COMMAND_GROUP = re.compile(r"\.\s*(submit)\s*\(")
K_QUEUE_SHORTCUT = re.compile(
    r"\b(?:q|q2|queue)\s*\.\s*(parallel_for|single_task|memcpy|memset|fill|copy)\s*\("
)
K_DEPENDS_ON_CALL = re.compile(r"\bdepends_on\s*\(")
K_DEPENDENCY_ARGUMENT = re.compile(r"\bdeps\w*\b")
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
class Module:
    """Holds one module folder and its tier."""

    tier: str
    name: str
    folder: Path


@dataclass(frozen=True, slots=True)
class ModuleTree:
    """Holds every module of one repository and the declared tier order."""

    root: Path
    modules: dict[str, Module]
    tier_order: tuple[str, ...]

    def rank(self, tier: str) -> int | None:
        """Returns a tier's position in the order, lowest first, or None when undeclared."""
        return self.tier_order.index(tier) if tier in self.tier_order else None


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds one repository root, its module tree when it has one, and its accepted violations."""

    root: Path
    tree: ModuleTree | None
    known: tuple[KnownViolation, ...]


@dataclass(frozen=True, slots=True)
class Violation:
    """Holds one layering violation found in a file, before the known table screens it."""

    path: Path
    line: int
    symbol: str


def _discover(root: Path) -> dict[str, Module]:
    """Returns every module below root/modules, keyed by module name."""
    modules: dict[str, Module] = {}
    base = root / "modules"
    for tier in sorted(entry for entry in base.iterdir() if entry.is_dir()):
        for folder in sorted(entry for entry in tier.iterdir() if entry.is_dir()):
            modules[folder.name] = Module(tier.name, folder.name, folder)
    return modules


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given repository folder with its module tree and the known table."""
    return [
        Repository(
            path,
            ModuleTree(path, _discover(path), K_TIER_ORDER) if (path / "modules").is_dir() else None,
            K_KNOWN_VIOLATIONS,
        )
        for path in paths if path.is_dir()
    ]


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


def _module_rule(rule: Callable[[ModuleTree], Iterator[Issue]]) -> Callable[[Repository], Iterator[Issue]]:
    """Returns a repository rule that runs a module-tree rule when the repository has modules/."""
    def apply(repository: Repository) -> Iterator[Issue]:
        if repository.tree is not None:
            yield from rule(repository.tree)
    apply.__doc__ = rule.__doc__
    return apply


def _declared_tiers(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each tier folder missing from the declared order."""
    if not tree.tier_order:
        yield Issue(Path(__file__), 1, "K_TIER_ORDER is empty; declare the repository's tiers")
        return
    for tier in sorted({module.tier for module in tree.modules.values()}):
        if tree.rank(tier) is None:
            yield Issue(tree.root / "modules" / tier, 1, f"tier '{tier}' is not in K_TIER_ORDER")


def _upward_includes(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include of a module that sits in a higher tier."""
    for module in tree.modules.values():
        own = tree.rank(module.tier)
        for path, number, line in _source_lines(module.folder):
            match = K_ANGLE_INCLUDE.match(line)
            target = tree.modules.get(match.group(1)) if match else None
            if own is None or target is None:
                continue
            other = tree.rank(target.tier)
            if other is not None and other > own:
                yield Issue(path, number, f"{module.tier}/{module.name} includes up into {target.tier}/{target.name}")


def _private_reach(tree: ModuleTree) -> Iterator[Issue]:
    """Yields an issue for each include or build line that reaches another module's source/."""
    for module in tree.modules.values():
        for path, number, line in _source_lines(module.folder):
            match = K_QUOTED_INCLUDE.match(line)
            if match and ("../" in match.group(1) or "source/" in match.group(1)):
                yield Issue(path, number, "quoted include leaves the module or names a source/ folder")
        build_file = module.folder / "CMakeLists.txt"
        if not build_file.is_file():
            continue
        for number, line in enumerate(build_file.read_text(encoding="utf-8").splitlines(), start=1):
            for name in K_CMAKE_SOURCE_REACH.findall(line):
                if name != module.name and name in tree.modules:
                    yield Issue(build_file, number, f"reaches into {name}/source")


rule_declared_tiers = _module_rule(_declared_tiers)
rule_upward_includes = _module_rule(_upward_includes)
rule_private_reach = _module_rule(_private_reach)


def _include_matches(lines: Iterable[tuple[Path, int, str]], pattern: re.Pattern[str]) -> Iterator[Violation]:
    """Yields a violation for each line whose include matches a pattern."""
    for path, number, line in lines:
        match = pattern.match(line)
        if match:
            yield Violation(path, number, match.group(1))


def find_kernel_includes(root: Path) -> Iterator[Violation]:
    """Yields each include of Engine, an engine/ header or the tensor header from a kernel."""
    for folder in K_KERNEL_FOLDERS:
        yield from _include_matches(_source_lines(root / folder), K_KERNEL_FORBIDDEN_INCLUDE)


def find_support_includes(root: Path) -> Iterator[Violation]:
    """Yields each include of Engine, engine/, kernels/ or the tensor header from support/."""
    yield from _include_matches(_source_lines(root / K_SUPPORT_FOLDER), K_SUPPORT_FORBIDDEN_INCLUDE)


def find_engine_header_includes(root: Path) -> Iterator[Violation]:
    """Yields each include from engine.hpp of a view, a seam or an experimental header."""
    for violation in _include_matches(_file_lines(root / K_ENGINE_HEADER), K_ENGINE_COMPOSED_INCLUDE):
        if violation.symbol not in K_ENGINE_BRICKS:
            yield violation


def _view_headers(root: Path) -> frozenset[str]:
    """Returns the operation views the umbrella header includes, as repository-relative paths."""
    views: set[str] = set()
    for _, _, line in _file_lines(root / K_UMBRELLA_HEADER):
        match = K_UMBRELLA_INCLUDE.match(line)
        if match and (match.group(1) == "io.hpp" or match.group(1).startswith("engine/")):
            views.add(f"{K_INCLUDE_FOLDER}/{match.group(1)}")
    return frozenset(views)


def find_seam_engine_includes(root: Path) -> Iterator[Violation]:
    """Yields each include of engine.hpp from an operation source or a header that is not a view."""
    exempt = _view_headers(root) | {K_ENGINE_HEADER, K_UMBRELLA_HEADER}
    headers = (
        entry for entry in _source_lines(root / K_INCLUDE_FOLDER)
        if _relative(entry[0], root) not in exempt
    )
    yield from _include_matches(headers, K_ENGINE_INCLUDE)
    yield from _include_matches(_source_lines(root / K_OPERATION_FOLDER), K_ENGINE_INCLUDE)


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
    yield from _engine_calls(_source_lines(root / K_OPERATION_FOLDER), K_OPERATION_ALLOWED_MEMBERS)


def find_view_engine_calls(root: Path) -> Iterator[Violation]:
    """Yields each call on an engine variable from a public header, other than get_recorder() and get_rng()."""
    headers = (
        entry for entry in _source_lines(root / K_INCLUDE_FOLDER)
        if _relative(entry[0], root) != K_ENGINE_HEADER
    )
    yield from _engine_calls(headers, K_VIEW_ALLOWED_MEMBERS)


def find_graph_insertions(root: Path) -> Iterator[Violation]:
    """Yields each add_task, add_node, add_native or add_offload_dynamic call outside TaskRecorder."""
    for folder in (K_INCLUDE_FOLDER, K_SOURCE_FOLDER):
        for path, number, line in _source_lines(root / folder):
            if K_LINE_COMMENT.match(line) or _relative(path, root) == K_RECORDER_HEADER:
                continue
            for call in K_GRAPH_INSERTION.findall(line):
                yield Violation(path, number, call)


def _call_text(lines: list[str], index: int, column: int) -> str:
    """Returns one call's text from the parenthesis at lines[index][column] to the one that closes it."""
    depth = 0
    text: list[str] = []
    for line in lines[index:]:
        for character in line[column:]:
            text.append(character)
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    return "".join(text)
        text.append("\n")
        column = 0
    return "".join(text)


def find_task_dependencies(root: Path) -> Iterator[Violation]:
    """Yields each command group without depends_on and each queue shortcut that names no deps."""
    wanted = ((K_COMMAND_GROUP, K_DEPENDS_ON_CALL), (K_QUEUE_SHORTCUT, K_DEPENDENCY_ARGUMENT))
    for folder in K_TASK_BODY_FOLDERS:
        for path, entries in groupby(_source_lines(root / folder), key=lambda entry: entry[0]):
            lines = [line for _, _, line in entries]
            for index, line in enumerate(lines):
                if K_LINE_COMMENT.match(line):
                    continue
                for call, evidence in wanted:
                    for match in call.finditer(line):
                        if not evidence.search(_call_text(lines, index, match.end() - 1)):
                            yield Violation(path, index + 1, match.group(1))


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
    "rule_support_independence": find_support_includes,
    "rule_kernel_includes": find_kernel_includes,
    "rule_engine_header_includes": find_engine_header_includes,
    "rule_seam_engine_includes": find_seam_engine_includes,
    "rule_operation_engine_calls": find_operation_engine_calls,
    "rule_view_engine_calls": find_view_engine_calls,
    "rule_graph_insertions": find_graph_insertions,
    "rule_task_dependencies": find_task_dependencies,
}

rule_support_independence = _screened_rule("rule_support_independence", find_support_includes)
rule_kernel_includes = _screened_rule("rule_kernel_includes", find_kernel_includes)
rule_engine_header_includes = _screened_rule("rule_engine_header_includes", find_engine_header_includes)
rule_seam_engine_includes = _screened_rule("rule_seam_engine_includes", find_seam_engine_includes)
rule_operation_engine_calls = _screened_rule("rule_operation_engine_calls", find_operation_engine_calls)
rule_view_engine_calls = _screened_rule("rule_view_engine_calls", find_view_engine_calls)
rule_graph_insertions = _screened_rule("rule_graph_insertions", find_graph_insertions)
rule_task_dependencies = _screened_rule("rule_task_dependencies", find_task_dependencies)

K_CHECKER = Checker(
    description="Checks module tiers and the support, kernel, Engine, view and recorder layers.",
    collect=collect,
    rules={
        "rule_declared_tiers": rule_declared_tiers,
        "rule_upward_includes": rule_upward_includes,
        "rule_private_reach": rule_private_reach,
        "rule_support_independence": rule_support_independence,
        "rule_kernel_includes": rule_kernel_includes,
        "rule_engine_header_includes": rule_engine_header_includes,
        "rule_seam_engine_includes": rule_seam_engine_includes,
        "rule_operation_engine_calls": rule_operation_engine_calls,
        "rule_view_engine_calls": rule_view_engine_calls,
        "rule_graph_insertions": rule_graph_insertions,
        "rule_task_dependencies": rule_task_dependencies,
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
