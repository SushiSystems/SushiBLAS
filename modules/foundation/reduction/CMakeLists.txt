# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME reduction
    LAYER foundation
    DEPENDS
        arithmetic
        launch)

sushiblas_add_test_program(
    MODULE reduction
    SIZE medium
    TESTS
        test_reduce_1d.cpp
        test_reduce_axis.cpp)
