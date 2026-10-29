# SushiBLASDebugInfo.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASDebugInfo — how much debug information the Debug build carries.
#
# Single responsibility: size the debug information of host code. Debug is the development
# build: host code unoptimised, with line tables only, which give a stack trace its files and
# lines at a fraction of full debug information's compile memory, object size and link time.
# CMake states -g for Debug; the later -gline-tables-only overrides it. Device code keeps its
# optimisation in every build type (SyclTarget.cmake). RelWithDebInfo carries full debug
# information. The rule is in the build-throughput skill of SushiSkills.

# -Xarch_host keeps the flag off a SYCL device pass, which rejects it.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:SHELL:-Xarch_host -gline-tables-only>")
endif()
