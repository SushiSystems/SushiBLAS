# test_tier_table.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that the tier table of the build and the layering checker agree with the module tree."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_layering import K_TIER_ORDER  # noqa: E402

K_REPOSITORY = Path(__file__).resolve().parents[2]
K_LAYERS_FILE = K_REPOSITORY / "cmake" / "SushiBLASLayers.cmake"
K_MODULES_FOLDER = K_REPOSITORY / "modules"


def _cmake_list(name: str) -> list[str]:
    """Returns the words of one set() call of the layers file."""
    text = K_LAYERS_FILE.read_text(encoding="utf-8")
    match = re.search(rf"^set\({name}\s+(.*?)\)", text, re.S | re.M)
    return match.group(1).split() if match else []


class TierTableTest(unittest.TestCase):
    """Checks the two statements of the tier order against each other and against the folders."""

    def test_the_checker_and_the_build_state_one_order(self) -> None:
        """The order of check_layering.py is the order of SushiBLASLayers.cmake."""
        self.assertEqual(list(K_TIER_ORDER), _cmake_list("SUSHIBLAS_LAYER_ORDER"))

    def test_every_module_folder_is_in_the_table_under_its_tier(self) -> None:
        """Each folder modules/<tier>/<module> is listed with that tier, and nothing else is."""
        words = _cmake_list("SUSHIBLAS_MODULE_LAYERS")
        listed = dict(zip(words[0::2], words[1::2]))
        found = {
            folder.name: folder.parent.name
            for folder in K_MODULES_FOLDER.glob("*/*") if folder.is_dir()
        }
        self.assertEqual(listed, found)


if __name__ == "__main__":
    unittest.main()
