# check_engine_seams.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks SushiBLAS's Engine seams: who includes engine.hpp, what a view calls, how a task orders.

The test-only and benchmark-only modules (INTERNAL, added only under SB_BUILD_TESTS or
SB_BUILD_BENCHMARKS) are exempt from these rules.

Usage: python tools/layering/check_engine_seams.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from functools import cache
from itertools import chain, groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, run  # noqa: E402
from layering.check_tier_contents import (  # noqa: E402
    K_ENGINE_HEADER,
    K_LINE_COMMENT,
    K_MODULES_FOLDER,
    K_OPERATION_TIER,
    Violation,
    _engine_calls,
    _header_lines,
    _include_matches,
    _module_folders,
    _operation_source_lines,
    _relative,
    _screened_rule,
    collect,
)

K_UMBRELLA_HEADER = "modules/library/umbrella/include/SushiBLAS/SushiBLAS.hpp"
K_ENGINE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](SushiBLAS/engine/engine\.hpp)[>"]')
K_VIEW_ALLOWED_MEMBERS = frozenset({"recorder", "rng"})
K_COMMAND_GROUP = re.compile(r"\.\s*(submit)\s*\(")
K_QUEUE_SHORTCUT = re.compile(
    r"\b(?:q|q2|queue)\s*\.\s*(parallel_for|single_task|memcpy|memset|fill|copy)\s*\("
)
K_DEPENDS_ON_CALL = re.compile(r"\bdepends_on\s*\(")
K_DEPENDENCY_ARGUMENT = re.compile(r"\bdeps\w*\b")
K_TESTS_SWITCH = re.compile(r"^\s*if\s*\(\s*SB_BUILD_(?:TESTS|BENCHMARKS)\s*\)", re.M)
K_ENDIF = re.compile(r"^\s*endif\s*\(", re.M)
K_SUBDIRECTORY = re.compile(r"add_subdirectory\(\s*([A-Za-z0-9_]+)\s*\)")
K_INTERNAL_FLAG = re.compile(r"^\s*INTERNAL\b", re.M)


@cache
def _test_only_prefixes(root: Path) -> tuple[str, ...]:
    """Returns the folder of each INTERNAL module a tier adds only under a test or benchmark switch."""
    prefixes: list[str] = []
    base = root / K_MODULES_FOLDER
    if not base.is_dir():
        return ()
    for tier in sorted(entry for entry in base.iterdir() if entry.is_dir()):
        build_file = tier / "CMakeLists.txt"
        if not build_file.is_file():
            continue
        text = build_file.read_text(encoding="utf-8")
        for switch in K_TESTS_SWITCH.finditer(text):
            end = K_ENDIF.search(text, switch.end())
            block = text[switch.end():end.start() if end else len(text)]
            for name in K_SUBDIRECTORY.findall(block):
                module_build = tier / name / "CMakeLists.txt"
                if module_build.is_file() and K_INTERNAL_FLAG.search(module_build.read_text(encoding="utf-8")):
                    prefixes.append(f"{K_MODULES_FOLDER}/{tier.name}/{name}/")
    return tuple(prefixes)


def _library_lines(root: Path, lines: Iterable[tuple[Path, int, str]]) -> Iterator[tuple[Path, int, str]]:
    """Yields the lines that do not belong to a test-only module."""
    exempt = _test_only_prefixes(root)
    for entry in lines:
        if not _relative(entry[0], root).startswith(exempt):
            yield entry


def _view_headers(root: Path) -> frozenset[str]:
    """Returns the view of each operation family, `<family>/<family>.hpp`, as repository-relative paths."""
    views: set[str] = set()
    for family in _module_folders(root, K_OPERATION_TIER):
        view = family / "include" / "SushiBLAS" / family.name / f"{family.name}.hpp"
        if view.is_file():
            views.add(_relative(view, root))
    return frozenset(views)


def find_seam_engine_includes(root: Path) -> Iterator[Violation]:
    """Yields each include of engine.hpp from an operation source or a header that is not a view."""
    exempt = _view_headers(root) | {K_ENGINE_HEADER, K_UMBRELLA_HEADER}
    headers = (
        entry for entry in _library_lines(root, _header_lines(root))
        if _relative(entry[0], root) not in exempt
    )
    yield from _include_matches(headers, K_ENGINE_INCLUDE)
    yield from _include_matches(_operation_source_lines(root), K_ENGINE_INCLUDE)


def find_view_engine_calls(root: Path) -> Iterator[Violation]:
    """Yields each call on an engine variable from a public header, other than recorder() and rng()."""
    headers = (
        entry for entry in _library_lines(root, _header_lines(root))
        if _relative(entry[0], root) != K_ENGINE_HEADER
    )
    yield from _engine_calls(headers, K_VIEW_ALLOWED_MEMBERS)


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
    bodies = chain(_library_lines(root, _header_lines(root)), _operation_source_lines(root))
    for path, entries in groupby(bodies, key=lambda entry: entry[0]):
        lines = [line for _, _, line in entries]
        for index, line in enumerate(lines):
            if K_LINE_COMMENT.match(line):
                continue
            for call, evidence in wanted:
                for match in call.finditer(line):
                    if not evidence.search(_call_text(lines, index, match.end() - 1)):
                        yield Violation(path, index + 1, match.group(1))


rule_seam_engine_includes = _screened_rule("rule_seam_engine_includes", find_seam_engine_includes)
rule_view_engine_calls = _screened_rule("rule_view_engine_calls", find_view_engine_calls)
rule_task_dependencies = _screened_rule("rule_task_dependencies", find_task_dependencies)

K_CHECKER = Checker(
    description="Checks who includes engine.hpp, what a view calls on an Engine and how a task orders its work.",
    collect=collect,
    rules={
        "rule_seam_engine_includes": rule_seam_engine_includes,
        "rule_view_engine_calls": rule_view_engine_calls,
        "rule_task_dependencies": rule_task_dependencies,
    },
)


if __name__ == "__main__":
    sys.exit(run(K_CHECKER))
