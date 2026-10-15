# test_label.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the labels `sb test --label` accepts and the selections they make."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass


class TestLabel(str, enum.Enum):
    """Names the labels `sb test --label` accepts; see cli/sushiblas/README.md."""

    __test__ = False

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    QUARANTINED = "quarantined"


#: The labels `sb test` runs when none is given: the sizes every pre-merge stage runs.
K_DEFAULT_LABELS = (TestLabel.SMALL, TestLabel.MEDIUM)


@dataclass(frozen=True, slots=True)
class LabelSelection:
    """Describes which tests a set of labels selects.

    Args:
        sizes: The sizes selected, in the order TestLabel declares them.
        quarantined: True when quarantined tests are selected as well.
    """

    sizes: tuple[TestLabel, ...]
    quarantined: bool

    def ctest_include(self) -> str:
        """Returns the `ctest -L` regex of this selection."""
        if not self.sizes:
            return f"^{TestLabel.QUARANTINED.value}$"
        return "^(" + "|".join(size.value for size in self.sizes) + ")$"

    def ctest_exclude(self) -> str | None:
        """Returns the `ctest -LE` regex of this selection, or None when it excludes nothing."""
        return None if self.quarantined else f"^{TestLabel.QUARANTINED.value}$"

    def pytest_expression(self) -> str:
        """Returns the `pytest -m` expression of this selection."""
        if not self.sizes:
            return TestLabel.QUARANTINED.value
        sizes = " or ".join(size.value for size in self.sizes)
        return f"({sizes})" if self.quarantined else f"({sizes}) and not quarantined"


def select_labels(labels: Iterable[TestLabel] | None) -> LabelSelection:
    """Returns the selection *labels* make; None or empty selects K_DEFAULT_LABELS."""
    chosen = set(labels or K_DEFAULT_LABELS)
    sizes = tuple(label for label in TestLabel
                  if label in chosen and label is not TestLabel.QUARANTINED)
    return LabelSelection(sizes=sizes, quarantined=TestLabel.QUARANTINED in chosen)
