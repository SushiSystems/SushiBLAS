# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Shared by the test fixture and the benchmark programs: a graph with a recorder on it, no test framework.
sushiblas_add_module(
    NAME recording_graph
    LAYER engine
    INTERNAL
    DEPENDS
        engine)

sushiblas_add_test_program(
    MODULE recording_graph
    DEVICE_TESTS
        test_recording_graph.cpp)
