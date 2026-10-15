# test_test_label.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that every label selection excludes quarantined tests unless it names them.

See cli/tests/README.md.
"""

from __future__ import annotations

import pytest

from sushiblas.services.test_label import TestLabel, select_labels

pytestmark = pytest.mark.small


def test_select_labels_defaults_to_small_and_medium_without_quarantined():
    """No label runs the small and medium tests and leaves the quarantined ones out."""
    selection = select_labels(None)
    assert selection.ctest_include() == "^(small|medium)$"
    assert selection.ctest_exclude() == "^quarantined$"
    assert selection.pytest_expression() == "(small or medium) and not quarantined"


def test_select_labels_keeps_the_declared_size_order():
    """The regex lists sizes in declaration order, whatever order the caller gave."""
    selection = select_labels([TestLabel.LARGE, TestLabel.SMALL])
    assert selection.ctest_include() == "^(small|large)$"


def test_select_labels_runs_only_quarantined_tests_when_only_they_are_named():
    """`--label quarantined` selects the quarantined tests and excludes nothing."""
    selection = select_labels([TestLabel.QUARANTINED])
    assert selection.ctest_include() == "^quarantined$"
    assert selection.ctest_exclude() is None
    assert selection.pytest_expression() == "quarantined"


def test_select_labels_includes_quarantined_tests_of_a_size_when_named():
    """A size with `quarantined` runs that size's tests, quarantined ones too."""
    selection = select_labels([TestLabel.SMALL, TestLabel.QUARANTINED])
    assert selection.ctest_include() == "^(small)$"
    assert selection.ctest_exclude() is None
    assert selection.pytest_expression() == "(small)"
