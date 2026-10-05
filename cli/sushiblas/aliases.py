# aliases.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""The old spellings `sb` still answers to, each with the day it stops working."""

from __future__ import annotations

from datetime import date

from sushicore.aliases import Alias, AliasTable

from .config import PROFILE

K_REMOVED_AFTER = date(2027, 3, 25)

K_ALIASES = AliasTable(PROFILE.program, (
    Alias("sb doxygen", "sb docs", K_REMOVED_AFTER),
))
