# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME gather
    LAYER operations
    DEVICE_SOURCES
        add_rows.cpp
        put_along.cpp
        take_along.cpp
        take_rows.cpp
    DEPENDS
        core
        tensor
        engine
    PRIVATE_DEPENDS
        arithmetic
        operands
        index_kernels)

sushiblas_add_test_program(
    MODULE gather
    TESTS
        test_index_operands.cpp
    DEVICE_TESTS
        test_gather/test_index_along.cpp
        test_gather/test_index_rows.cpp
    DEVICE_BENCHMARKS
        benchmark_index_rows.cpp)
