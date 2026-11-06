# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME reference
    LAYER foundation
    DEPENDS
        core
        reduction)

sushiblas_add_test_program(
    MODULE reference
    TESTS
        test_reference_accumulator.cpp
        test_serial_reduce.cpp
    DEVICE_TESTS
        test_serial_1d.cpp)
