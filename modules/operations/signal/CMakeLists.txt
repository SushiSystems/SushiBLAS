# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME signal
    LAYER operations
    DEVICE_SOURCES
        convolve_1d.cpp
    DEPENDS
        tensor
        engine
    PRIVATE_DEPENDS
        signal_kernels)

sushiblas_add_test_program(
    MODULE signal
    TESTS
        test_convolve_operand_check.cpp
    DEVICE_TESTS
        test_signal/test_signal_convolve_1d.cpp)
