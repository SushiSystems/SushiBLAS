# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

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
        test_cost_shape.cpp
        test_cost_table.cpp)
