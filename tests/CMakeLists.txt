# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The tests that cross module boundaries. A test of one module lives in that module's
# tests/ folder and is built by its CMakeLists.txt; both go through
# sushiblas_add_test_program() (cmake/SushiBLASTests.cmake), so they are labelled and
# linked the same way. See tests/README.md.
sushiblas_add_test_program(
    NAME sushiblas_tests
    SOURCES
        unit/test_channel_bias_epilogue.cpp
        unit/test_dispatch_complex.cpp
        unit/test_dispatch_seam.cpp
        unit/test_empty_tensors.cpp
        unit/test_functor_reference.cpp
        unit/test_integer_dtypes.cpp
        unit/test_math_validation.cpp
        unit/test_packed_product.cpp
        unit/test_reference_mode_guard.cpp
        unit/test_runtime_shapes.cpp
        unit/test_scratch_buffer.cpp
        unit/test_sparse_builder.cpp
        unit/test_sparse_types.cpp
        unit/test_spatial_random_io_validation.cpp
        unit/test_structured_operands.cpp
        integration/test_dependency_stress.cpp
        integration/test_framework_sim.cpp
        regression/test_blas_symmetric_bits.cpp
        regression/test_io.cpp
        regression/test_logger.cpp
        regression/test_operand_lifetime.cpp
        regression/test_regression.cpp
        regression/test_sparse_recording.cpp
        regression/test_spatial_determinism.cpp
        regression/test_task_dependencies.cpp)
