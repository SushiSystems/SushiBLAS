# check_operation_costs.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every operation id the library records has a row in the cost table and no row is stale.

An operation id is a `"name"_op` literal under modules/. A row is a `costed("name"_op, ...)` or
`uncosted("name"_op, ...)` call in the cost table source.

Usage: python tools/profiling/check_operation_costs.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_MODULES_FOLDER = "modules"
K_COST_MODULE = Path("modules/foundation/cost")
K_COST_TABLE = K_COST_MODULE / "source" / "cost_table.cpp"
K_SOURCE_SUFFIXES = frozenset({".hpp", ".cpp"})
K_OPERATION_LITERAL = re.compile(r'"([A-Za-z0-9_.]+)"_op\b')
K_COST_ROW = re.compile(r'\b(?:costed|uncosted)\(\s*"([A-Za-z0-9_.]+)"_op\b')


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the root of a repository whose operation ids and cost table the rules compare."""

    root: Path

    def operation_sites(self) -> dict[str, tuple[Path, int]]:
        """Returns the first place each operation id is written outside the cost module."""
        sites: dict[str, tuple[Path, int]] = {}
        base = self.root / K_MODULES_FOLDER
        if not base.is_dir():
            return sites
        skipped = self.root / K_COST_MODULE
        for path in sorted(base.rglob("*")):
            if path.suffix not in K_SOURCE_SUFFIXES or skipped in path.parents:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                for name in K_OPERATION_LITERAL.findall(line):
                    sites.setdefault(name, (path, number))
        return sites

    def row_sites(self) -> dict[str, int]:
        """Returns the line of each row of the cost table."""
        table = self.root / K_COST_TABLE
        if not table.is_file():
            return {}
        rows: dict[str, int] = {}
        for number, line in enumerate(table.read_text(encoding="utf-8").splitlines(), 1):
            for name in K_COST_ROW.findall(line):
                rows.setdefault(name, number)
        return rows


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns one repository for each given folder."""
    return [Repository(path) for path in paths if path.is_dir()]


def rule_operation_without_row(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each operation id the library records that the cost table has no row for."""
    rows = repository.row_sites()
    for name, (path, line) in sorted(repository.operation_sites().items()):
        if name not in rows:
            yield Issue(path, line, f'operation "{name}" has no row in {K_COST_TABLE.as_posix()}')


def rule_row_without_operation(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each row of the cost table whose operation id no code records."""
    operations = repository.operation_sites()
    table = repository.root / K_COST_TABLE
    for name, line in sorted(repository.row_sites().items(), key=lambda item: item[1]):
        if name not in operations:
            yield Issue(table, line, f'row "{name}" names an operation no module records')


K_CHECKER = Checker(
    description="Checks that the cost table has a row for every recorded operation, and no other.",
    collect=collect,
    rules={
        "rule_operation_without_row": rule_operation_without_row,
        "rule_row_without_operation": rule_row_without_operation,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))
