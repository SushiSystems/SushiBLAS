# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME signal
    LAYER operations
    SOURCES
        convolve_1d.cpp
    DEPENDS
        tensor
        engine
    PRIVATE_DEPENDS
        signal_kernels)

sushiblas_add_test_program(
    MODULE signal
    TESTS
        test_signal/test_signal_convolve_1d.cpp)
