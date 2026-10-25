# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME random_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        launch
        reduction
        reference
        orthonormal)

sushiblas_add_test_program(
    MODULE random_kernels
    TESTS
        reference/test_orthonormalize.cpp
        test_random.cpp
    DEVICE_TESTS
        sycl/test_orthonormalize.cpp
        sycl/test_philox.cpp
        sycl/test_poisson.cpp
        sycl/test_truncated_normal.cpp)
