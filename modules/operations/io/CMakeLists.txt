# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME io
    LAYER operations
    SOURCES
        file_header.cpp
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
        test_file_header.cpp
        test_io_status.cpp
        test_partial_file.cpp
    DEVICE_TESTS
        test_io/test_host_transfer.cpp
        test_io/test_regression.cpp
    COMPILE_CHECKS
        test_io_log_category.cpp)
