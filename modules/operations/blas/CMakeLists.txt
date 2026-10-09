# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME blas
    LAYER operations
    SOURCES
        level1/asum.cpp
        level1/axpy.cpp
        level1/copy.cpp
        level1/dot.cpp
        level1/iamax.cpp
        level1/nrm2.cpp
        level1/rot.cpp
        level1/rotg.cpp
        level1/rotm.cpp
        level1/rotmg.cpp
        level1/scal.cpp
        level1/swap.cpp
        level2/gbmv.cpp
        level2/gemv.cpp
        level2/ger.cpp
        level2/sbmv.cpp
        level2/spmv.cpp
        level2/spr.cpp
        level2/spr2.cpp
        level2/symv.cpp
        level2/syr.cpp
        level2/syr2.cpp
        level2/tbmv.cpp
        level2/tbsv.cpp
        level2/tpmv.cpp
        level2/tpsv.cpp
        level2/trmv.cpp
        level2/trsv.cpp
        level3/gemm.cpp
        level3/symm.cpp
        level3/syr2k.cpp
        level3/syrk.cpp
        level3/trsm.cpp
    DEPENDS
        core
        support
        tensor
        blas_seam_kernels
        blas_kernels
        engine
        operands)

sushiblas_add_module_tests(
    MODULE blas
    SOURCES
        test_axpy.cpp
        test_blas_band.cpp
        test_blas_complex.cpp
        test_blas_half_accumulation.cpp
        test_blas_packed.cpp
        test_blas_structured_validation.cpp
        test_blas_validation.cpp
        test_copy.cpp
        test_dot.cpp
        test_gemm.cpp
        test_gemv.cpp
        test_ger.cpp
        test_iamax.cpp
        test_level1_parallel_reductions.cpp
        test_level1_reference.cpp
        test_level2_reference.cpp
        test_level2_sycl_parity.cpp
        test_packed_symm.cpp
        test_packed_syrk.cpp
        test_rot.cpp
        test_rotg.cpp
        test_rotm.cpp
        test_rotmg.cpp
        test_scal.cpp
        test_swap.cpp
        test_symm.cpp
        test_symv.cpp
        test_syr.cpp
        test_syr2.cpp
        test_syr2k.cpp
        test_syrk.cpp
        test_syrk_trsm_parity.cpp
        test_triangular_parallel.cpp
        test_triangular_plan.cpp
        test_trmv.cpp
        test_trsm.cpp
        test_trsm_reference.cpp
        test_trsv.cpp)
