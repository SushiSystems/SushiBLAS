# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME spatial_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        arithmetic
        launch
        reduction
        reference
        geometry)

sushiblas_add_test_program(
    MODULE spatial_kernels
    TESTS
        reference/test_broadcast.cpp
        reference/test_channels.cpp
        reference/test_col2im.cpp
        reference/test_im2col.cpp
        reference/test_pool_avg.cpp
        reference/test_pool_max.cpp
        reference/test_upsample.cpp
        reference/test_window_dot.cpp
        reference/test_window_dot_weights.cpp
        sycl/test_broadcast.cpp
        sycl/test_channels.cpp
        sycl/test_col2im.cpp
        sycl/test_im2col.cpp
        sycl/test_pool_avg.cpp
        sycl/test_pool_max.cpp
        sycl/test_upsample.cpp
        sycl/test_window_dot.cpp
        sycl/test_window_dot_weights.cpp
        test_spatial.cpp)
