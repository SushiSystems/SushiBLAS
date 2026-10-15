# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the BLAS truth, the flag mappings and the band and packed conversions.
sushiblas_add_module(
    NAME blas_test_oracle
    LAYER foundation
    INTERNAL
    DEPENDS
        core)

sushiblas_add_test_program(
    MODULE blas_test_oracle
    TESTS
        test_blas_flags.cpp
        test_blas_structured.cpp
        test_blas_truth.cpp)
