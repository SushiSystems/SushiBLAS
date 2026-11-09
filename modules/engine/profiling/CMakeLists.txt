# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Built only with SB_ENABLE_PROFILER (modules/engine/CMakeLists.txt).
sushiblas_add_module(
    NAME profiling
    LAYER engine
    SOURCES
        blas_environment.cpp
        ceilings_file.cpp
        folder_ceilings_source.cpp
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
        test_ceilings_file.cpp
        test_folder_ceilings_source.cpp
        test_in_memory_ceilings_source.cpp
    DEVICE_TESTS
        test_profiler_parts.cpp
    COMPILE_CHECKS
        compile_ceilings.cpp
        compile_ceilings_source.cpp
        compile_profiling_log_category.cpp)
