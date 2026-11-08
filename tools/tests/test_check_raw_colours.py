# test_check_raw_colours.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the raw colour rule over Python sources held in memory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from styling.check_raw_colours import K_CHECKER, PythonSource, rule_raw_colour_strings  # noqa: E402

K_TEST_SIZE = "small"
K_PATH = Path("cli/sushiblas/cli.py")


def _issues(text: str) -> list[Issue]:
    """Returns what the rule reports for one file of *text*."""
    return list(rule_raw_colour_strings(PythonSource(K_PATH, text)))


class RawColourStringsTest(unittest.TestCase):
    """Checks which string literals the rule reports."""

    def test_is_registered(self) -> None:
        """The checker has the one rule."""
        self.assertEqual(list(K_CHECKER.rules), ["rule_raw_colour_strings"])

    def test_reports_an_opening_colour_tag_on_its_line(self) -> None:
        """A string that opens [green] is reported on the line it starts."""
        self.assertEqual(
            _issues('x = 1\nprint("[green]done[/green]")\n'),
            [Issue(K_PATH, 2, "names the raw colour '[green]'; use a theme style")],
        )

    def test_reports_a_closing_colour_tag_inside_an_f_string(self) -> None:
        """The literal part of an f-string is read, so a closing [/red] is reported."""
        self.assertEqual(
            _issues('name = "x"\ntext = f"{name}[/red]"\n'),
            [Issue(K_PATH, 2, "names the raw colour '[/red]'; use a theme style")],
        )

    def test_accepts_a_theme_style(self) -> None:
        """A theme style such as [muted] is not a raw colour."""
        self.assertEqual(_issues('print("[muted]path[/muted]")\n'), [])

    def test_accepts_a_word_that_only_starts_with_a_colour(self) -> None:
        """[greenish] names no colour the theme bypasses."""
        self.assertEqual(_issues('print("[greenish]")\n'), [])

    def test_reports_a_file_that_does_not_parse(self) -> None:
        """A file that is not Python is reported, not skipped."""
        self.assertEqual(_issues("def (:\n"), [Issue(K_PATH, 1, "cannot be parsed: invalid syntax")])


if __name__ == "__main__":
    unittest.main()
