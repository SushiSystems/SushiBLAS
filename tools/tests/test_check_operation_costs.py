# test_check_operation_costs.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the cost table checker: an operation without a row, and a row without an operation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.check_operation_costs import (  # noqa: E402
    K_CHECKER,
    Repository,
    rule_operation_without_row,
    rule_row_without_operation,
)

K_TABLE = "modules/foundation/cost/source/cost_table.cpp"
K_SEAM = "modules/operations/blas/source/level3/gemm.cpp"


def _repository(folder: str, files: dict[str, str]) -> Repository:
    """Returns a repository at folder that holds the given files."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root)


class RegistrationTest(unittest.TestCase):
    """Checks the rule table of the cost checker."""

    def test_is_registered(self) -> None:
        """Lists the two rules."""
        self.assertEqual(list(K_CHECKER.rules), ["rule_operation_without_row", "rule_row_without_operation"])


class OperationWithoutRowTest(unittest.TestCase):
    """Checks that every recorded operation id has a row."""

    def test_accepts_an_operation_with_a_row(self) -> None:
        """Reports nothing when the seam's literal has a costed row."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {
                K_SEAM: 'description.operation = "blas.lvl3.gemm"_op;\n',
                K_TABLE: 'costed("blas.lvl3.gemm"_op, 0, 0, 0),\n',
            })
            self.assertEqual(list(rule_operation_without_row(repository)), [])

    def test_accepts_an_operation_with_an_uncosted_row(self) -> None:
        """Reports nothing when the row declares a reason."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {
                K_SEAM: 'x = "blas.lvl3.gemm"_op;\n',
                K_TABLE: 'uncosted("blas.lvl3.gemm"_op, UncostedReason::Unaudited),\n',
            })
            self.assertEqual(list(rule_operation_without_row(repository)), [])

    def test_reports_an_operation_with_no_row(self) -> None:
        """Reports the seam line of a literal the table does not list."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {
                K_SEAM: '\nx = "blas.lvl3.gemm"_op;\n',
                K_TABLE: 'costed("blas.lvl1.copy"_op, 0, 0, 0),\n',
            })
            issues = list(rule_operation_without_row(repository))
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].line, 2)
            self.assertIn("blas.lvl3.gemm", issues[0].message)

    def test_ignores_a_literal_inside_the_cost_module(self) -> None:
        """Does not count the table's own literals as recorded operations."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_TABLE: 'costed("blas.lvl3.gemm"_op, 0, 0, 0),\n'})
            self.assertEqual(list(rule_operation_without_row(repository)), [])


class RowWithoutOperationTest(unittest.TestCase):
    """Checks that no row outlives its operation."""

    def test_reports_a_stale_row(self) -> None:
        """Reports the table line of a row no module records."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {
                K_SEAM: 'x = "blas.lvl3.gemm"_op;\n',
                K_TABLE: 'costed("blas.lvl3.gemm"_op, 0, 0, 0),\ncosted("gone.op"_op, 1, 0, 0),\n',
            })
            issues = list(rule_row_without_operation(repository))
            self.assertEqual([issue.line for issue in issues], [2])
            self.assertIn("gone.op", issues[0].message)

    def test_accepts_a_row_with_an_operation(self) -> None:
        """Reports nothing when every row has its literal."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {
                K_SEAM: 'x = "blas.lvl3.gemm"_op;\n',
                K_TABLE: 'costed("blas.lvl3.gemm"_op, 0, 0, 0),\n',
            })
            self.assertEqual(list(rule_row_without_operation(repository)), [])


if __name__ == "__main__":
    unittest.main()
