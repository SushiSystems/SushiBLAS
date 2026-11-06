# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME cost
    LAYER foundation
    SOURCES
        cost_table.cpp
    DEPENDS
        core
    EXTERNAL
        SushiRuntime::SushiRuntime)

sushiblas_add_test_program(
    MODULE cost
    TESTS
        test_composite_cost.cpp
        test_cost.cpp
        test_cost_table.cpp
    COMPILE_CHECKS
        compile_cost_shape.cpp)
