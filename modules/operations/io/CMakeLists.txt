# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME io
    LAYER operations
    SOURCES
        io.cpp
        print.cpp
    DEPENDS
        core
        tensor
    PRIVATE_DEPENDS
        engine)

sushiblas_add_test_program(
    MODULE io
    TESTS
        test_io_log_category.cpp
        test_io_status.cpp
    DEVICE_TESTS
        test_io/test_host_transfer.cpp
        test_io/test_regression.cpp)
