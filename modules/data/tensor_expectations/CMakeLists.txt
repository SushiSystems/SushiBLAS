# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: writing host values into a dense tensor and comparing it with host expectations.
sushiblas_add_module(
    NAME tensor_expectations
    LAYER data
    INTERNAL
    DEPENDS
        core
        tensor
        test_runner
    EXTERNAL
        GTest::gtest)

sushiblas_add_test_program(
    MODULE tensor_expectations
    TESTS
        test_scalar_dtype.cpp
        test_tensor_comparison.cpp
        test_tensor_fill.cpp
        test_tensor_readback.cpp)
