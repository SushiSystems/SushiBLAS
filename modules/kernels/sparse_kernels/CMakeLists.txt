# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME sparse_kernels
    LAYER kernels
    INTERNAL
    DEPENDS
        core
        arithmetic
        launch
        reduction
        reference
        substitution
        blas_seam_kernels
        blas_kernels)

sushiblas_add_test_program(
    MODULE sparse_kernels
    TESTS
        reference/test_level1.cpp
        reference/test_spmm.cpp
        reference/test_spmv.cpp
        reference/test_spsymv.cpp
        reference/test_sptrsv.cpp
        reference/test_structure.cpp
        sycl/test_level1.cpp
        sycl/test_row_blocks.cpp
        sycl/test_spmm.cpp
        sycl/test_spmv.cpp
        sycl/test_spsymv.cpp
        sycl/test_sptrsv.cpp
        test_csr_pattern.cpp
        test_index_list.cpp
        test_sparse.cpp)
