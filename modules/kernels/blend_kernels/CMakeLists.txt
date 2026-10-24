# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

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
        reference/test_blend_transforms.cpp
        sycl/test_blend_transforms.cpp
        test_blend.cpp)
