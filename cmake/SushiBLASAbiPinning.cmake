# SushiBLASAbiPinning.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASAbiPinning — the C runtime and the STL layout every target of this tree compiles
# against, in every build type.
#
# Single responsibility: make a Debug tree link what a Release tree links. SushiRuntime and
# SushiLog arrive as Release packages, so a debug C runtime or a checked STL here would put two
# heaps and two container layouts in one process. Debug is the development build: unoptimised
# host code against the same dependencies as Release, with no debug build of the stack beneath
# it. Memory errors are the sanitizer tree's job. The rule is in the build-throughput skill of
# SushiSkills.

# The runtime selection only takes effect under the NEW behaviour; refuse rather than pin nothing.
cmake_policy(GET CMP0091 _sb_runtime_policy)
if(NOT _sb_runtime_policy STREQUAL "NEW")
    message(FATAL_ERROR
        "SushiBLASAbiPinning: policy CMP0091 is '${_sb_runtime_policy}', so "
        "CMAKE_MSVC_RUNTIME_LIBRARY would be ignored. Raise cmake_minimum_required to 3.15 "
        "or later.")
endif()
unset(_sb_runtime_policy)

# One shared release C runtime for every build type; ignored off the MSVC ABI.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# A Debug consumer of an imported target links its Release build, which shares that runtime.
set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release RelWithDebInfo MinSizeRel None "")

# The MSVC STL's container layout depends on this; libstdc++ and libc++ never read it.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX>:_ITERATOR_DEBUG_LEVEL=0>")
endif()
