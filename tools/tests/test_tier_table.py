# test_tier_table.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that the tier table of the build and the layering checker agree with the module tree."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import K_TIER_ORDER  # noqa: E402

K_REPOSITORY = Path(__file__).resolve().parents[2]
K_LAYERS_FILE = K_REPOSITORY / "cmake" / "SushiBLASLayers.cmake"
K_MODULES_FOLDER = K_REPOSITORY / "modules"


def _cmake_list_words(text: str, name: str) -> list[str]:
    """Returns the words of the set() call called name in text, or an empty list when it has none."""
    match = re.search(rf"^set\({name}\s+(.*?)\)", text, re.S | re.M)
    return match.group(1).split() if match else []


def _layers_words(name: str) -> list[str]:
    """Returns the words of one set() call of the layers file."""
    return _cmake_list_words(K_LAYERS_FILE.read_text(encoding="utf-8"), name)


class CmakeListWordsTest(unittest.TestCase):
    """Checks the reader of a set() call, on text and not on the layers file."""

    K_TEST_SIZE = "small"

    def test_reads_the_words_of_a_call_over_several_lines(self) -> None:
        """Returns every word between the name and the closing parenthesis."""
        text = "set(OTHER x)\nset(ORDER\n    foundation\n    engine)\n"
        self.assertEqual(_cmake_list_words(text, "ORDER"), ["foundation", "engine"])

    def test_returns_nothing_for_a_name_the_text_does_not_set(self) -> None:
        """Returns an empty list when no call sets the name."""
        self.assertEqual(_cmake_list_words("set(OTHER x)\n", "ORDER"), [])


class TierTableTest(unittest.TestCase):
    """Checks the two statements of the tier order against each other and against the folders."""

    K_TEST_SIZE = "medium"

    def test_the_checker_and_the_build_state_one_order(self) -> None:
        """The order of tools/settings.py is the order of SushiBLASLayers.cmake."""
        self.assertEqual(list(K_TIER_ORDER), _layers_words("SUSHIBLAS_LAYER_ORDER"))

    def test_every_module_folder_is_in_the_table_under_its_tier(self) -> None:
        """Each folder modules/<tier>/<module> is listed with that tier, and nothing else is."""
        words = _layers_words("SUSHIBLAS_MODULE_LAYERS")
        listed = dict(zip(words[0::2], words[1::2]))
        found = {
            folder.name: folder.parent.name
            for folder in K_MODULES_FOLDER.glob("*/*") if folder.is_dir()
        }
        self.assertEqual(listed, found)


if __name__ == "__main__":
    unittest.main()
