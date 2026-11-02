# test_next_jobs.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the output and the exit codes of the next-jobs command."""

from __future__ import annotations

import contextlib
import io
import pytest
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.next_jobs import main  # noqa: E402

K_HEAD = "## Jobs\n\n| Job | Files | Waits on | Status |\n| --- | --- | --- | --- |\n"


pytestmark = pytest.mark.medium


def run(rows: str, free: int = 1) -> tuple[int, str]:
    """Runs the command on a plan holding the rows and returns the exit code and the output."""
    with tempfile.TemporaryDirectory() as folder:
        plan = Path(folder) / "PLAN.md"
        plan.write_text(K_HEAD + rows, encoding="utf-8")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = main([str(plan), "--free", str(free)])
    return code, captured.getvalue()


class NextJobsTest(unittest.TestCase):
    """Covers each of the four exit codes and the printed ids."""

    def test_prints_the_ready_jobs_and_exits_zero(self) -> None:
        code, output = run("| A | a | - | todo |\n| B | b | - | todo |\n", free=2)
        self.assertEqual((code, output), (0, "A\nB\n"))

    def test_exits_one_when_jobs_remain_and_none_is_ready(self) -> None:
        code, output = run("| A | a | - | running |\n| B | b | A | todo |\n")
        self.assertEqual((code, output), (1, ""))

    def test_exits_two_when_every_job_is_done(self) -> None:
        self.assertEqual(run("| A | a | - | done |\n"), (2, ""))

    def test_exits_three_on_a_malformed_table_and_names_the_line(self) -> None:
        code, output = run("| A | a | - | later |\n")
        self.assertEqual(code, 3)
        self.assertIn("PLAN.md:5: status:", output)

    def test_exits_three_on_a_cycle(self) -> None:
        code, output = run("| A | a | B | todo |\n| B | b | A | todo |\n")
        self.assertEqual(code, 3)
        self.assertIn("cycle", output)


if __name__ == "__main__":
    unittest.main()
