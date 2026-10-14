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

sushiblas_add_module_tests(
    MODULE reduction
    SOURCES
        test_reduce_plans.cpp
        test_reduce_segments.cpp)
