# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME orthonormal
    LAYER foundation)

sushiblas_add_test_program(
    MODULE orthonormal
    SIZE small
    TESTS
        test_orthonormal_sweep.cpp)
