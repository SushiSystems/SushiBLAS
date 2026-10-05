# SushiBLASCompilerWarnings.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASCompilerWarnings.cmake — the project's warning policy as a target.
#
# Single responsibility: define an INTERFACE target `sushiblas_warnings`
# carrying the compiler warning level and the preprocessor definitions a
# warning-clean build needs. Architecture targeting lives in Optimization.cmake
# because it answers to a different owner: warnings are a review standard,
# architecture is a deployment choice.
#
# BuildPolicy.cmake links this PRIVATE, so `-Werror` is a rule for this tree
# and never something an installed consumer inherits — mirrors
# sushiruntime/cmake/CompilerWarnings.cmake exactly.

add_library(sushiblas_warnings INTERFACE)

if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
    # Windows / MSVC / Intel icx-cl
    target_compile_options(sushiblas_warnings INTERFACE
        /W4 /permissive-
    )
    target_compile_definitions(sushiblas_warnings INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
    # Linux / GCC / Clang / Intel oneAPI
    target_compile_options(sushiblas_warnings INTERFACE
        -Wall
        -Wextra
        -Werror
        -pedantic
    )
endif()
