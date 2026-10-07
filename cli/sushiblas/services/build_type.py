# build_type.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the build types `sb build --type` accepts and their CMake spellings."""

from __future__ import annotations

import enum


class BuildType(str, enum.Enum):
    """Names the build types `sb build --type` accepts."""

    RELEASE = "release"
    DEBUG = "debug"
    RELWITHDEBINFO = "relwithdebinfo"


#: Maps each build type to the CMAKE_BUILD_TYPE value it configures.
K_CMAKE_BUILD_TYPE = {
    BuildType.RELEASE: "Release",
    BuildType.DEBUG: "Debug",
    BuildType.RELWITHDEBINFO: "RelWithDebInfo",
}
