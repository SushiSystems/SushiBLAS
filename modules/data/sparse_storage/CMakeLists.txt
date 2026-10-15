# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME sparse_storage
    LAYER data
    SOURCES
        sparse_builder.cpp
    DEPENDS
        core
        sparse_assembly
        tensor)

sushiblas_add_test_program(
    MODULE sparse_storage
    SIZE medium
    TESTS
        test_sparse_builder.cpp
        test_sparse_matrix.cpp
        test_sparse_vector.cpp)
