# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Built only with SB_ENABLE_PROFILER (modules/engine/CMakeLists.txt).
sushiblas_add_module(
    NAME profiling
    LAYER engine
    SOURCES
        blas_environment.cpp
        ceilings_file.cpp
        profiler_parts.cpp
    DEPENDS
        core
    PRIVATE_DEPENDS
        launch
        profile
    EXTERNAL
        SushiProfiler::SushiProfiler)

sushiblas_add_test_program(
    MODULE profiling
    TESTS
        test_blas_environment.cpp
        test_ceilings.cpp
        test_ceilings_file.cpp
        test_profiling_log_category.cpp
    DEVICE_TESTS
        test_profiler_parts.cpp)
