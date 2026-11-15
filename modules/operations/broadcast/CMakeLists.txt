# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME broadcast
    LAYER operations
    DEVICE_SOURCES
        add.cpp
        div.cpp
        mul.cpp
        sub.cpp
    DEPENDS
        core
        tensor
        engine
        functor_map)

sushiblas_add_test_program(
    MODULE broadcast
    DEVICE_TESTS
        test_broadcast.cpp
        test_dispatch.cpp
    COMPILE_CHECKS
        compile_broadcast_checks.cpp
        compile_broadcast_operand.cpp
    DEVICE_BENCHMARKS
        benchmark_broadcast.cpp)
