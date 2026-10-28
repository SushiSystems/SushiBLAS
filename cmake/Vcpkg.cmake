# Vcpkg.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Vcpkg — (Windows) locate the vcpkg root and point CMAKE_TOOLCHAIN_FILE at it,
# before project(). SushiBLAS is the top-level project, so it reaches project()
# first; the runtime it pulls in needs hwloc from vcpkg on Windows, and resolving
# the toolchain file here — ahead of that subproject — is what lets that resolve.
# No-op off Windows.
if(NOT WIN32)
    return()
endif()

# Prefer an explicit VCPKG_ROOT (env or cache). `sb build` passes -DVCPKG_ROOT itself, from the
# vcpkg tree `sb setup` found, so a workspace layout is read by the CLI and not here.
if((NOT DEFINED VCPKG_ROOT OR VCPKG_ROOT STREQUAL "") AND DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_ROOT "$ENV{VCPKG_ROOT}" CACHE PATH "Path to vcpkg root")
endif()

if(VCPKG_ROOT AND (NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL ""))
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain file")
endif()
