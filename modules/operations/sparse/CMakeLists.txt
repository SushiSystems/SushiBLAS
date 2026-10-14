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
        launch
        tensor
        sparse_storage
        blas_seam_kernels
        sparse_kernels
        engine
        operands)

sushiblas_add_module_tests(
    MODULE sparse
    SOURCES
        test_sparse_level1.cpp
        test_sparse_spmm.cpp
        test_sparse_spmv.cpp
        test_sparse_spsymv.cpp
        test_sparse_sptrsv.cpp
        test_sparse_validation.cpp)
