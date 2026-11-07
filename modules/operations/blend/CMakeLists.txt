# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

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
    TESTS
        test_blend_operand_check.cpp
    DEVICE_TESTS
        test_blend/test_blend_transforms.cpp)
