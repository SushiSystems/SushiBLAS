# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME blas_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        core
        arithmetic
        launch
        reduction
        reference
        substitution
        blas_seam_kernels)

sushiblas_add_test_program(
    MODULE blas_kernels
    TESTS
        test_blas_kernels_log_category.cpp
        reference/level1/test_level1.cpp
        reference/level1/test_rotg.cpp
        reference/level1/test_rotm.cpp
        reference/level1/test_rotmg.cpp
        reference/level2/test_gbmv.cpp
        reference/level2/test_gemv.cpp
        reference/level2/test_ger.cpp
        reference/level2/test_sbmv.cpp
        reference/level2/test_spmv.cpp
        reference/level2/test_spr.cpp
        reference/level2/test_spr2.cpp
        reference/level2/test_symmetric_product.cpp
        reference/level2/test_symv.cpp
        reference/level2/test_syr.cpp
        reference/level2/test_syr2.cpp
        reference/level2/test_tbmv.cpp
        reference/level2/test_tbsv.cpp
        reference/level2/test_tpmv.cpp
        reference/level2/test_tpsv.cpp
        reference/level2/test_triangular_product.cpp
        reference/level2/test_triangular_solve.cpp
        reference/level2/test_trmv.cpp
        reference/level3/test_syr2k.cpp
        reference/level3/test_syrk.cpp
        reference/level3/test_trsm.cpp
        sycl/level1/test_level1.cpp
        sycl/level1/test_rotg.cpp
        sycl/level1/test_rotm.cpp
        sycl/level1/test_rotmg.cpp
        sycl/level2/test_gbmv.cpp
        sycl/level2/test_gemv.cpp
        sycl/level2/test_ger.cpp
        sycl/level2/test_row_reduction.cpp
        sycl/level2/test_sbmv.cpp
        sycl/level2/test_spmv.cpp
        sycl/level2/test_spr.cpp
        sycl/level2/test_spr2.cpp
        sycl/level2/test_symmetric_product.cpp
        sycl/level2/test_symv.cpp
        sycl/level2/test_syr.cpp
        sycl/level2/test_syr2.cpp
        sycl/level2/test_tbmv.cpp
        sycl/level2/test_tbsv.cpp
        sycl/level2/test_tpmv.cpp
        sycl/level2/test_tpsv.cpp
        sycl/level2/test_triangular_product.cpp
        sycl/level2/test_triangular_solve.cpp
        sycl/level2/test_trmv.cpp
        sycl/level2/test_vector_snapshot.cpp
        sycl/level3/test_syr2k.cpp
        sycl/level3/test_syrk.cpp
        sycl/level3/test_triangular_plan.cpp
        sycl/level3/test_triangular_system.cpp
        sycl/level3/test_trsm.cpp
        test_blas.cpp)
