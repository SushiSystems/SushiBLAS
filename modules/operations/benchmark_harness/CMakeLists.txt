# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Benchmark-only: the argument, environment, fill, timing and table helpers every benchmark
# program shares. It is added only when SB_BUILD_BENCHMARKS is on, and then with
# SB_ENABLE_PROFILER (modules/operations/CMakeLists.txt): its profile session records through the
# profiling module.

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
        recording_graph
        profile
        profiling)

sushiblas_add_test_program(
    MODULE benchmark_harness
    TESTS
        test_benchmark_harness.cpp
        test_environment_value.cpp
        test_profile_session.cpp)
