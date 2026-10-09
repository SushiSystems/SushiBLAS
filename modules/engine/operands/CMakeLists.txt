# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME operands
    LAYER engine
    DEPENDS
        core
        support
        tensor
        engine)

sushiblas_add_module_tests(
    MODULE operands
    SOURCES
        test_blas_scalar.cpp)
