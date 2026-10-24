# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Shared by the test fixture and the benchmark programs: a graph with a recorder on it, no test framework.
sushiblas_add_module(
    NAME recording_graph
    LAYER engine
    INTERNAL
    DEPENDS
        engine)

sushiblas_add_test_program(
    MODULE recording_graph
    TESTS
        test_recording_graph.cpp)
