# Runtime.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Runtime — D3 layered resolution of the SushiRuntime dependency.
#
# SushiBLAS no longer vendors a snapshot of SushiRuntime (SB-0): it consumes the
# live checkout through exactly the three modes D3 defines, tried in order, so
# that whichever one resolves, every consumer links the identical namespaced
# alias afterwards:
#   1. TARGET-guard — a superproject that already brought SushiRuntime in
#      (e.g. a future AI-0 tree building sushiblas and sushiruntime together).
#   2. find_package(SushiRuntime QUIET) — an installed package (BB-4's
#      install/export set), the out-of-tree case.
#   3. add_subdirectory of a sibling checkout — today's actual case on this
#      machine: no installed package exists yet, so this is what fires.
# sushiruntime's own src/CMakeLists.txt unconditionally defines the
# SushiRuntime::SushiRuntime ALIAS regardless of how it's brought in, so all
# three modes converge on the same link name with no extra plumbing here.
if(NOT TARGET SushiRuntime::SushiRuntime)
    find_package(SushiRuntime QUIET)
endif()

if(NOT TARGET SushiRuntime::SushiRuntime)
    # Inside a SushiStack workspace every module is cloned beside the others,
    # so the runtime sibling is <workspace>/sushiruntime. Prefer
    # SUSHISTACK_HOME when set (the installer exports it); otherwise the
    # in-tree sibling ../sushiruntime, which is the same path in a workspace
    # and the right standalone default (mirrors sushiengine/cmake/Runtime.cmake).
    if(DEFINED ENV{SUSHISTACK_HOME} AND EXISTS "$ENV{SUSHISTACK_HOME}/sushiruntime")
        set(_SB_DEFAULT_RUNTIME "$ENV{SUSHISTACK_HOME}/sushiruntime")
    else()
        set(_SB_DEFAULT_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/../sushiruntime")
    endif()
    set(SUSHIRUNTIME_DIR "${_SB_DEFAULT_RUNTIME}"
        CACHE PATH "Path to the SushiRuntime sibling checkout")

    add_subdirectory("${SUSHIRUNTIME_DIR}" "${CMAKE_BINARY_DIR}/sushiruntime")
endif()
