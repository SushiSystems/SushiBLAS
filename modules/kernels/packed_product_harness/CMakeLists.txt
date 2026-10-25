# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the problem, the data and the blockings of the packed product tests.
sushiblas_add_module(
    NAME packed_product_harness
    LAYER kernels
    INTERNAL
    DEPENDS
        core
        arithmetic
        launch
        blas_seam_kernels)

sushiblas_add_test_program(
    MODULE packed_product_harness
    DEVICE_TESTS
        test_packed_product_harness.cpp)
