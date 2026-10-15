# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME sparse
    LAYER operations
    SOURCES
        axpyi.cpp
        doti.cpp
        gthr.cpp
        gthrz.cpp
        roti.cpp
        sctr.cpp
        spmm.cpp
        spmv.cpp
        spsymv.cpp
        sptrsv.cpp
    DEPENDS
        core
        arithmetic
        tensor
        sparse_storage
        engine
        operands
    PRIVATE_DEPENDS
        blas_seam_kernels
        launch
        sparse_kernels)

sushiblas_add_test_program(
    MODULE sparse
    SIZE medium
    TESTS
        test_accumulator_size.cpp
        test_index_check.cpp
        test_sparse/test_sparse_level1.cpp
        test_sparse/test_sparse_recording.cpp
        test_sparse/test_sparse_spmm.cpp
        test_sparse/test_sparse_spmv.cpp
        test_sparse/test_sparse_spsymv.cpp
        test_sparse/test_sparse_sptrsv.cpp
        test_sparse/test_sparse_validation.cpp
        test_sparse_matrix_operand.cpp
        test_sparse_vector_operand.cpp
    BENCHMARKS
        benchmark_sparse.cpp)
