# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME linalg_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        arithmetic
        launch)

sushiblas_add_test_program(
    MODULE linalg_kernels
    TESTS
        reference/test_lu.cpp
        reference/test_gmres.cpp
        test_chunked_dot.cpp
        test_gmres_workspace.cpp
    DEVICE_TESTS
        sycl/test_lu.cpp
        sycl/test_gmres.cpp
        test_linalg.cpp)
