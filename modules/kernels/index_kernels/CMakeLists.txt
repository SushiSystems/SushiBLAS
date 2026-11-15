# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME index_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        arithmetic
        launch
        reference
        substitution)

sushiblas_add_test_program(
    MODULE index_kernels
    TESTS
        test_spans.cpp
        reference/test_along.cpp
        reference/test_rows.cpp
    DEVICE_TESTS
        sycl/test_along.cpp
        sycl/test_rows.cpp
    COMPILE_CHECKS
        compile_index.cpp)
