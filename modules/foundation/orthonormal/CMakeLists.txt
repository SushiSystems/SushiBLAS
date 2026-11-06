# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME orthonormal
    LAYER foundation)

sushiblas_add_test_program(
    MODULE orthonormal
    COMPILE_CHECKS
        compile_orthonormal_sweep.cpp)
