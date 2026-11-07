# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME blend_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        arithmetic
        launch
        reference)

sushiblas_add_test_program(
    MODULE blend_kernels
    TESTS
        test_blend_index.cpp
        reference/test_blend_transforms.cpp
    DEVICE_TESTS
        sycl/test_blend_transforms.cpp
    COMPILE_CHECKS
        compile_blend.cpp)
