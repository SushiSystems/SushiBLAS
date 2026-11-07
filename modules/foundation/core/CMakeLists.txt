# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME core
    LAYER foundation
    EXTERNAL
        SushiLog::SushiLog
        SushiLog::text)

sushiblas_add_test_program(
    MODULE core
    TESTS
        test_sushiblas_log_project.cpp
        test_blas_types.cpp
        test_common.cpp
        test_errors.cpp
        test_export.cpp
        test_logger.cpp
        test_memory_log_port.cpp
        test_sparse_types.cpp
        test_sushilog_port.cpp
        test_traits.cpp
        test_version.cpp
    COMPILE_CHECKS
        compile_blas_types.cpp
        compile_common.cpp
        compile_errors.cpp
        compile_log_port.cpp
        compile_sparse_types.cpp
        compile_sushiblas_log_project.cpp
        compile_traits.cpp)
