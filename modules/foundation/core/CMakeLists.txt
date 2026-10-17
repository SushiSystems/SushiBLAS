# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

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
        test_sparse_types.cpp
        test_traits.cpp
        test_version.cpp)
