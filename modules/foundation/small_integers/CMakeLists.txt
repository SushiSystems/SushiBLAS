# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Test-only: the integer-valued operands a kernel test fills, so both kernels compute exactly.
sushiblas_add_module(
    NAME small_integers
    LAYER foundation
    INTERNAL)

sushiblas_add_test_program(
    MODULE small_integers
    TESTS
        test_small_integers.cpp)
