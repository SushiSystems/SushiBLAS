# test_suites.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the Suite enum and the one suite that is not a CTest label.

`sb test` maps every suite to a `ctest -L` regex through _SUITE_LABEL_REGEX,
with two deliberate exceptions - `all` (no label selection at all) and
`package` (no ctest at all). Both are absent from the mapping, so both depend
on `test()` branching before it indexes it. A missing branch is a KeyError at
the moment the user runs the command, which these tests exist to prevent.
"""

from __future__ import annotations

from sushiblas.services.project import _SUITE_LABEL_REGEX, Suite


def test_every_suite_is_either_mapped_or_deliberately_special():
    """No suite may be silently unmapped: it is either a label or a known exception."""
    special = {Suite.all, Suite.package}
    for suite in Suite:
        assert (suite in _SUITE_LABEL_REGEX) ^ (suite in special), (
            f"{suite.value!r} is neither a CTest label nor a handled special case; "
            "add it to _SUITE_LABEL_REGEX or branch on it in test()")


def test_package_is_not_a_ctest_label():
    """Indexing the mapping with `package` must not accidentally start working.

    If someone gives `package` a label regex, `test()` would still branch away
    to test_package() first and the label would be dead code that reads as live.
    """
    assert Suite.package not in _SUITE_LABEL_REGEX


def test_all_does_not_select_the_package_suite():
    """`all` runs every CTest test, and package is not one of them.

    Pinned because the alternative - quietly folding package into `all` - would
    make a full run several minutes slower and, worse, make `all` mean two
    different things depending on whether the tree happened to be installable.
    """
    assert Suite.all not in _SUITE_LABEL_REGEX
    assert all(
        "package" not in regex for regex in _SUITE_LABEL_REGEX.values()
    ), "no CTest label group may claim to cover the package consumer"
