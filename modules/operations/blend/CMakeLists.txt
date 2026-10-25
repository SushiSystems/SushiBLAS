# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME blend
    LAYER operations
    DEVICE_SOURCES
        blend_transforms.cpp
    DEPENDS
        tensor
        engine
    PRIVATE_DEPENDS
        blend_kernels)

sushiblas_add_test_program(
    MODULE blend
    DEVICE_TESTS
        test_blend/test_blend_transforms.cpp)
