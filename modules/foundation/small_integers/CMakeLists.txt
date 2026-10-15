# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the integer-valued operands a kernel test fills, so both kernels compute exactly.
sushiblas_add_module(
    NAME small_integers
    LAYER foundation
    INTERNAL)

sushiblas_add_test_program(
    MODULE small_integers
    SIZE small
    TESTS
        test_small_integers.cpp)
