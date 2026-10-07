# suite.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the test suites `sb test --suite` accepts and their CTest label regexes."""

from __future__ import annotations

import enum


class Suite(str, enum.Enum):
    """Names the test suites `sb test --suite` accepts; see cli/sushiblas/README.md."""

    UNIT = "unit"
    REGRESSION = "regression"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    ALL = "all"
    PACKAGE = "package"


#: Maps each CTest-backed suite to its `ctest -L` regex; see cli/sushiblas/README.md.
K_SUITE_LABEL_REGEX = {
    Suite.UNIT: "^unit$",
    Suite.REGRESSION: "^regression$",
    Suite.INTEGRATION: "^integration$",
    Suite.FUNCTIONAL: "^(unit|integration|regression)$",
}
