# SushiBLASOptimization.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASOptimization.cmake — architecture targeting as a target.
#
# Single responsibility: define an INTERFACE target `sushiblas_perf` carrying
# the code-generation flags that are a property of the machine a binary is
# built *for*. The optimization level itself is not here: CMake's own
# CMAKE_CXX_FLAGS_<CONFIG> already supplies -O3 / /O2 in Release.
#
# Architecture targeting is opt-in. `-march=native`/`/arch:AVX2` bakes the
# build machine's exact ISA into the binary, which is right for a developer
# benchmarking on the box in front of them and wrong for anything shipped: the
# resulting binary faults with an illegal instruction on any older CPU.
# BuildPolicy.cmake links this PRIVATE, so it applies to this tree only and
# never reaches an installed consumer — mirrors
# sushiruntime/cmake/Optimization.cmake (minus the TLS-model flag, which is
# specific to a SHARED library with thread-local allocator state; sushiblas is
# STATIC and has no equivalent).

option(SB_NATIVE_ARCH
    "Build for the build machine's exact ISA (-march=native / /arch:AVX2). Developer convenience; must stay OFF for any binary that leaves this machine."
    OFF)

add_library(sushiblas_perf INTERFACE)

if(SB_NATIVE_ARCH)
    if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
        target_compile_options(sushiblas_perf INTERFACE /arch:AVX2)
    else()
        target_compile_options(sushiblas_perf INTERFACE -march=native)
    endif()
    message(STATUS "SB_NATIVE_ARCH=ON — binaries target this machine's ISA and are not portable.")
endif()
