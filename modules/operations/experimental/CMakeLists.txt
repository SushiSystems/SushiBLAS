# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME experimental
    LAYER operations
    DEVICE_SOURCES
        fft.cpp
        gmres.cpp
        lu_factor.cpp
        lu_solve.cpp
    DEPENDS
        tensor
        engine
    PRIVATE_DEPENDS
        core
        operands
        signal_kernels
        linalg_kernels)

sushiblas_add_test_program(
    MODULE experimental
    TESTS
        test_dense_operand.cpp
    DEVICE_TESTS
        lapack/test_linalg.cpp
        signal/test_transforms.cpp)
