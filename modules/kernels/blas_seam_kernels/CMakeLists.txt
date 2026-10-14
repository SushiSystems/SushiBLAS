# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME blas_seam_kernels
    LAYER kernels
    DEPENDS
        core
        arithmetic
        launch
        reduction
        reference)

sushiblas_add_module_tests(
    MODULE blas_seam_kernels
    SOURCES
        test_band_access.cpp
        test_givens.cpp
        test_operand_access.cpp
        test_packed_access.cpp
        test_packed_micro_kernel.cpp
        test_packed_panel.cpp
        test_packed_product_plan.cpp
        test_symmetric_access.cpp
        test_symmetric_row.cpp)
