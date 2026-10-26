# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME arithmetic
    LAYER foundation)

sushiblas_add_test_program(
    MODULE arithmetic
    TESTS
        test_accumulator.cpp
        test_complex_arithmetic.cpp
        test_multiply_add.cpp
        test_no_contraction.cpp
        test_device_math.cpp
        test_safe_math.cpp)
