# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME signal_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        arithmetic
        launch
        reference)

sushiblas_add_test_program(
    MODULE signal_kernels
    TESTS
        reference/test_convolve_1d.cpp
        reference/test_fft.cpp
        test_convolve_index.cpp
        test_fft_line_index.cpp
        test_fft_plan.cpp
    DEVICE_TESTS
        sycl/test_convolve_1d.cpp
        sycl/test_fft.cpp
    COMPILE_CHECKS
        compile_signal.cpp)
