# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME spatial
    LAYER operations
    SOURCES
        broadcast.cpp
        channels.cpp
        col2im.cpp
        im2col.cpp
        pool_avg.cpp
        pool_max.cpp
        pool_max_backward.cpp
        upsample_nearest.cpp
        window_dot.cpp
        window_dot_weights_backward.cpp
    DEPENDS
        core
        launch
        reduction
        geometry
        tensor
        spatial_kernels
        engine)

sushiblas_add_test_program(
    MODULE spatial
    TESTS
        test_spatial/test_spatial_broadcast.cpp
        test_spatial/test_spatial_channels.cpp
        test_spatial/test_spatial_col2im.cpp
        test_spatial/test_spatial_determinism.cpp
        test_spatial/test_spatial_half.cpp
        test_spatial/test_spatial_im2col.cpp
        test_spatial/test_spatial_pool_avg.cpp
        test_spatial/test_spatial_pool_max.cpp
        test_spatial/test_spatial_upsample.cpp
        test_spatial/test_spatial_window_dot.cpp)
