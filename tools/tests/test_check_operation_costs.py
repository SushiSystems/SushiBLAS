# test_check_operation_costs.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the cost table checker: an operation without a row, and a row without an operation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.check_operation_costs import (  # noqa: E402
    K_CHECKER,
    MemoryRepository,
    Repository,
    operation_sites_in,
    row_sites_in,
    rule_operation_without_row,
    rule_row_without_operation,
)

K_TABLE = "modules/foundation/cost/source/cost_table.cpp"
K_SEAM = "modules/operations/blas/source/level3/gemm.cpp"
K_SEAM_PATH = Path(K_SEAM)
K_COST_MODULE = Path("modules/foundation/cost")
K_GEMM = "blas.lvl3.gemm"


class RegistrationTest(unittest.TestCase):
    """Checks the rule table of the cost checker."""

    K_TEST_SIZE = "small"

    def test_is_registered(self) -> None:
        """Lists the two rules."""
        self.assertEqual(list(K_CHECKER.rules), ["rule_operation_without_row", "rule_row_without_operation"])


class TestOperationWithoutRow:
    """Checks that every recorded operation id has a row."""

    K_TEST_SIZE = "small"

    @pytest.mark.parametrize("row", [
        pytest.param('costed("blas.lvl3.gemm"_op, 0, 0, 0),\n', id="costed"),
        pytest.param('uncosted("blas.lvl3.gemm"_op, UncostedReason::Unaudited),\n', id="uncosted"),
    ])
    def test_accepts_an_operation_with_a_row(self, row: str) -> None:
        """Reports nothing when the seam's literal has a costed or an uncosted row."""
        repository = MemoryRepository(sites={K_GEMM: (K_SEAM_PATH, 1)}, rows=row_sites_in(row))
        assert list(rule_operation_without_row(repository)) == []

    def test_reports_an_operation_with_no_row(self) -> None:
        """Reports the seam line of a literal the table does not list."""
        repository = MemoryRepository(
            sites={K_GEMM: (K_SEAM_PATH, 2)},
            rows=row_sites_in('costed("blas.lvl1.copy"_op, 0, 0, 0),\n'),
        )
        issues = list(rule_operation_without_row(repository))
        assert [(issue.path, issue.line) for issue in issues] == [(K_SEAM_PATH, 2)]
        assert issues[0].message == f'operation "{K_GEMM}" has no row in {K_TABLE}'

    def test_finds_the_first_literal_of_an_id_in_the_given_files(self) -> None:
        """Returns the file and line of the first literal of each id."""
        files = [(K_SEAM_PATH, f'\nx = "{K_GEMM}"_op;\ny = "{K_GEMM}"_op;\n')]
        assert operation_sites_in(files, K_COST_MODULE) == {K_GEMM: (K_SEAM_PATH, 2)}

    def test_ignores_a_literal_inside_the_cost_module(self) -> None:
        """Does not count the table's own literals as recorded operations."""
        files = [(Path(K_TABLE), f'costed("{K_GEMM}"_op, 0, 0, 0),\n')]
        assert operation_sites_in(files, K_COST_MODULE) == {}


class TestRowWithoutOperation:
    """Checks that no row outlives its operation."""

    K_TEST_SIZE = "small"

    def test_reports_a_stale_row(self) -> None:
        """Reports the table line of a row no module records."""
        repository = MemoryRepository(
            sites={K_GEMM: (K_SEAM_PATH, 1)},
            rows=row_sites_in(f'costed("{K_GEMM}"_op, 0, 0, 0),\ncosted("gone.op"_op, 1, 0, 0),\n'),
        )
        issues = list(rule_row_without_operation(repository))
        assert [(issue.path, issue.line) for issue in issues] == [(Path(".") / K_TABLE, 2)]
        assert issues[0].message == 'row "gone.op" names an operation no module records'

    def test_accepts_a_row_with_an_operation(self) -> None:
        """Reports nothing when every row has its literal."""
        repository = MemoryRepository(
            sites={K_GEMM: (K_SEAM_PATH, 1)},
            rows=row_sites_in(f'costed("{K_GEMM}"_op, 0, 0, 0),\n'),
        )
        assert list(rule_row_without_operation(repository)) == []


class RepositoryReadsTheFilesTest(unittest.TestCase):
    """Checks that the repository reads the seam and the table from disk."""

    K_TEST_SIZE = "medium"

    def test_reads_a_seam_and_a_table_from_a_folder(self) -> None:
        """Reports the seam line of a literal the table on disk does not list."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in {
                K_SEAM: f'\nx = "{K_GEMM}"_op;\n',
                K_TABLE: 'costed("blas.lvl1.copy"_op, 0, 0, 0),\n',
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            issues = list(rule_operation_without_row(Repository(root)))
            self.assertEqual([(issue.path, issue.line) for issue in issues], [(root / K_SEAM, 2)])


if __name__ == "__main__":
    unittest.main()
