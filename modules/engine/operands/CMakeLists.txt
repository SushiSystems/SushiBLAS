# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME operands
    LAYER engine
    DEPENDS
        core
        cost
        launch
        reduction
        substitution
        tensor
        engine)

sushiblas_add_test_program(
    MODULE operands
    TESTS
        test_operands_log_category.cpp
        test_argument_check.cpp
        test_blas_metadata.cpp
        test_matrix_operand.cpp
        test_scalar.cpp
        test_scalar_result.cpp
        test_scratch_lease.cpp
        test_vector_operand.cpp)
