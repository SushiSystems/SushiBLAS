# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Benchmark-only: the argument, environment, fill, timing and table helpers every benchmark
# program shares. It is added only when SB_BUILD_BENCHMARKS is on.
sushiblas_add_module(
    NAME benchmark_harness
    LAYER operations
    INTERNAL
    DEPENDS
        core
        launch
        tensor
        engine
        io
        recording_graph)

sushiblas_add_test_program(
    MODULE benchmark_harness
    TESTS
        test_benchmark_harness.cpp)
