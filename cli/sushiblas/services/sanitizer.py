# sanitizer.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the sanitizer sets `sb build --sanitize` accepts and their SB_SANITIZE values."""

from __future__ import annotations

import enum

#: The CMake cache variable cmake/SushiBLASSanitizers.cmake reads.
K_SANITIZE_VARIABLE = "SB_SANITIZE"


class Sanitizer(str, enum.Enum):
    """Names the sanitizer sets `sb build --sanitize` accepts."""

    NONE = "none"
    ADDRESS_UNDEFINED = "address-undefined"
    THREAD = "thread"


#: Maps each sanitizer set to the SB_SANITIZE value it configures.
K_SANITIZE_VALUE = {
    Sanitizer.NONE: "",
    Sanitizer.ADDRESS_UNDEFINED: "address,undefined",
    Sanitizer.THREAD: "thread",
}


def sanitize_flags(sanitizer: Sanitizer | None) -> list[str]:
    """Returns the -D flag that sets SB_SANITIZE; None leaves the cached value."""
    if sanitizer is None:
        return []
    return [f"-D{K_SANITIZE_VARIABLE}={K_SANITIZE_VALUE[sanitizer]}"]
