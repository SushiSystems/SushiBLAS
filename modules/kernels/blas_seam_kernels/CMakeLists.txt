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

sushiblas_add_test_program(
    MODULE blas_seam_kernels
    TESTS
        level3/test_channel_bias_epilogue.cpp
        level3/test_gemm_epilogue.cpp
        level3/test_packed_product_plan.cpp
        reference/level3/test_gemm.cpp
        sycl/level3/test_gemm.cpp
        sycl/level3/test_gemm_accessed_packed.cpp
        sycl/level3/test_gemm_packed.cpp
        sycl/level3/test_packed_micro_kernel.cpp
        sycl/level3/test_packed_panel.cpp
        sycl/level3/test_packed_product.cpp
        test_band_access.cpp
        test_conjugation.cpp
        test_givens.cpp
        test_modified_givens.cpp
        test_operand_access.cpp
        test_packed_access.cpp
        test_row_cursor.cpp
        test_symmetric_access.cpp
        test_symmetric_row.cpp)
