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
        test_complex_arithmetic.cpp)
