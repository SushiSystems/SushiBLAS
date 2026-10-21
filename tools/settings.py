# settings.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Holds the settings of one repository that the shared checkers read.

copy_shared_tools.py writes this file only when a repository has none, and never overwrites it.
"""

from __future__ import annotations

K_TIER_ORDER: tuple[str, ...] = ("foundation", "data", "kernels", "engine", "operations", "library")
K_DOMAIN_TIERS: tuple[str, ...] = ()
K_SAME_TIER_EDGES: frozenset[tuple[str, str]] = frozenset()
K_MODULE_FUNCTION = "sushiblas_add_module"
K_PUBLIC_DEPENDS = "DEPENDS"
K_PRIVATE_DEPENDS = "PRIVATE_DEPENDS"
K_INTERNAL_MARK = "INTERNAL"
K_PRIVATE_DEPENDS_ALSO: tuple[str, ...] = ()
K_CLOSED = False
K_MACRO_PREFIXES: tuple[str, ...] = ()
K_LOG_PREFIX = "SB"
# The benchmark harness prints its result table and IO::print prints a tensor: both are output.
K_CONSOLE_PATHS: tuple[str, ...] = (
    "modules/operations/benchmark_harness",
    "modules/operations/io/source/print.cpp",
    "cli/sushiblas",
)
K_DESIGN_TOPICS = True
K_FOLDER_CEILING = 60
K_CLOSED_GRACE_DAYS = 0
K_ARCHIVE_UNTRACKED = True
