# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# ============================================================================
# SushiBLAS test pipeline
#
# One binary, sushiblas_tests, covering unit/, integration/ and
# regression/. Tests are grouped into CTest labels (unit / integration /
# regression) from the GTest suite-name prefix convention (Unit_* /
# Integration_* / Regression_*), matching sushiruntime/sushiengine, so
# `ctest -L unit` (or the umbrella regex the CLI uses) selects a sub-suite and
# new tests are picked up automatically as long as they follow the convention.
# See tests/README.md.
# ============================================================================

cmake_minimum_required(VERSION 3.22) # gtest_discover_tests(TEST_FILTER ...)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

set(SB_TEST_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_executable(sushiblas_tests
    common/test_main.cpp

    # --- Level 1 ---
    unit/test_asum.cpp
    unit/test_axpy.cpp
    unit/test_copy.cpp
    unit/test_dot.cpp
    unit/test_iamax.cpp
    unit/test_nrm2.cpp
    unit/test_rot.cpp
    unit/test_scal.cpp
    unit/test_swap.cpp

    # --- Level 2 ---
    unit/test_gemv.cpp
    unit/test_ger.cpp
    unit/test_symv.cpp
    unit/test_syr.cpp
    unit/test_syr2.cpp
    unit/test_trmv.cpp
    unit/test_trsv.cpp

    # --- Level 3 ---
    unit/test_gemm.cpp
    unit/test_syrk.cpp
    unit/test_trsm.cpp

    # --- BLAS value types, bricks and the complex routines ---
    unit/test_blas_types.cpp
    unit/test_blas_scalar.cpp
    unit/test_complex_arithmetic.cpp
    unit/test_operand_access.cpp
    unit/test_running_solution.cpp
    unit/test_reference_accumulator.cpp
    unit/test_blas_complex.cpp

    # --- Level 3 completion, band, packed, rotations and the packed GEMM path ---
    unit/test_symmetric_access.cpp
    unit/test_symm.cpp
    unit/test_syr2k.cpp
    unit/test_triangular_plan.cpp
    unit/test_triangular_parallel.cpp
    unit/test_givens.cpp
    unit/test_rotg.cpp
    unit/test_rotm.cpp
    unit/test_rotmg.cpp
    unit/test_band_access.cpp
    unit/test_packed_access.cpp
    unit/test_symmetric_row.cpp
    unit/test_structured_operands.cpp
    unit/test_blas_structured_validation.cpp
    unit/test_blas_band.cpp
    unit/test_blas_packed.cpp
    unit/test_packed_product_plan.cpp
    unit/test_packed_panel.cpp
    unit/test_packed_micro_kernel.cpp
    unit/test_packed_product.cpp
    unit/test_packed_syrk.cpp
    unit/test_packed_symm.cpp

    # --- Spatial (windowed movement over an N-D grid) ---
    unit/test_spatial_geometry.cpp
    unit/test_spatial_im2col.cpp
    unit/test_spatial_col2im.cpp
    unit/test_spatial_pool_max.cpp
    unit/test_spatial_pool_avg.cpp
    unit/test_spatial_upsample.cpp
    unit/test_spatial_channels.cpp
    unit/test_spatial_broadcast.cpp
    unit/test_mean_spatial.cpp
    unit/test_spatial_half.cpp
    unit/test_spatial_window_dot.cpp
    unit/test_channel_bias_epilogue.cpp

    # --- Nonlinear ---
    unit/test_relu.cpp
    unit/test_leaky_relu.cpp
    unit/test_sigmoid.cpp
    unit/test_tanh.cpp
    unit/test_elu.cpp
    unit/test_silu.cpp
    unit/test_gelu.cpp
    unit/test_softplus.cpp

    # --- Math: Random ---
    unit/test_constant.cpp
    unit/test_distributions.cpp
    unit/test_initializers.cpp
    unit/test_normal.cpp
    unit/test_orthogonal.cpp
    unit/test_shuffle.cpp
    unit/test_uniform.cpp

    # --- Dispatch seam (the public SushiBLAS::Dispatch generic executors) ---
    unit/test_dispatch_complex.cpp
    unit/test_dispatch_seam.cpp
    unit/test_read_binding.cpp
    unit/test_functor_reference.cpp
    unit/test_random_reference.cpp

    # --- Tensor views (every op family against a non-zero storage_offset) ---
    unit/test_storage_offset.cpp
    unit/test_tensor_view.cpp

    # --- Element types (which ops serve INT32/INT64, and which refuse them) ---
    unit/test_integer_dtypes.cpp

    # --- Elementwise ---
    unit/test_abs.cpp
    unit/test_acos.cpp
    unit/test_acosh.cpp
    unit/test_add.cpp
    unit/test_asin.cpp
    unit/test_asinh.cpp
    unit/test_atan.cpp
    unit/test_atanh.cpp
    unit/test_ceil.cpp
    unit/test_clamp.cpp
    unit/test_convert.cpp
    unit/test_cos.cpp
    unit/test_cosh.cpp
    unit/test_div.cpp
    unit/test_exp.cpp
    unit/test_floor.cpp
    unit/test_fmod.cpp
    unit/test_log.cpp
    unit/test_max.cpp
    unit/test_min.cpp
    unit/test_mul.cpp
    unit/test_neg.cpp
    unit/test_pow.cpp
    unit/test_reciprocal.cpp
    unit/test_remainder.cpp
    unit/test_round.cpp
    unit/test_sin.cpp
    unit/test_sinh.cpp
    unit/test_sqrt.cpp
    unit/test_square.cpp
    unit/test_sub.cpp
    unit/test_tan.cpp

    # --- Reductions ---
    unit/test_reductions.cpp
    unit/test_reduction_determinism.cpp
    unit/test_axis_reductions.cpp
    unit/test_count_nonfinite.cpp
    unit/test_reduce_plans.cpp
    unit/test_reduce_segments.cpp

    # --- Logic ---
    unit/test_all.cpp
    unit/test_any.cpp
    unit/test_equal.cpp
    unit/test_greater.cpp
    unit/test_greater_equal.cpp
    unit/test_less.cpp
    unit/test_less_equal.cpp
    unit/test_logical_and.cpp
    unit/test_logical_not.cpp
    unit/test_logical_or.cpp
    unit/test_logical_xor.cpp
    unit/test_not_equal.cpp
    unit/test_where.cpp

    # --- Operand validation and execution modes ---
    unit/test_operand_check.cpp
    unit/test_execution_mode.cpp
    unit/test_unsupported_execution.cpp
    unit/test_blas_validation.cpp
    unit/test_math_validation.cpp
    unit/test_spatial_random_io_validation.cpp

    # --- Reference kernels and SYCL parity ---
    unit/test_level1_reference.cpp
    unit/test_level2_reference.cpp
    unit/test_trsm_reference.cpp
    unit/test_syrk_trsm_parity.cpp
    unit/test_level1_parallel_reductions.cpp
    unit/test_level2_sycl_parity.cpp

    # --- Empty tensors, integer arithmetic, host transfer, runtime shapes ---
    unit/test_empty_tensors.cpp
    unit/test_integer_arithmetic.cpp
    unit/test_host_transfer.cpp
    unit/test_runtime_shapes.cpp
    unit/test_random_large_lambda.cpp
    unit/test_sum_edge_cases.cpp
    unit/test_max_min_nan.cpp
    unit/test_complex_nonlinear.cpp
    unit/test_blas_half_accumulation.cpp
    unit/test_reference_mode_guard.cpp
    unit/test_tensor_const_access.cpp
    unit/test_rng_stream.cpp

    # --- Support layer and scratch ---
    unit/test_scratch_shelf.cpp
    unit/test_launch_shapes.cpp
    unit/test_scratch_buffer.cpp
    unit/test_serial_loops.cpp

    # --- Recorder, handles and the tensor factory ---
    unit/test_task_handle.cpp
    unit/test_task_recorder.cpp
    unit/test_tensor_factory.cpp

    # --- Sparse ---
    unit/test_sparse_types.cpp
    unit/test_sparse_structure.cpp
    unit/test_sparse_builder.cpp
    unit/test_sparse_assembly.cpp
    unit/test_sparse_validation.cpp
    unit/test_sparse_level1.cpp
    unit/test_sparse_spmv.cpp
    unit/test_sparse_spsymv.cpp
    unit/test_sparse_spmm.cpp
    unit/test_sparse_sptrsv.cpp

    # --- Integration ---
    integration/test_framework_sim.cpp
    integration/test_dependency_stress.cpp

    # --- Regression ---
    regression/test_regression.cpp
    regression/test_spatial_determinism.cpp
    regression/test_io.cpp
    regression/test_logger.cpp
    regression/test_operand_lifetime.cpp
    regression/test_task_dependencies.cpp
    regression/test_blas_symmetric_bits.cpp
    regression/test_sparse_recording.cpp
)

# --- Include Directories ---
# source/ is here for the same reason the library has it: it is what makes
# `#include <kernels/...>` resolve. Tests need it to reach a kernel variant the
# public Engine API cannot select on its own - GEMM chooses between its
# local-tiled and no-local shapes from the device profile, so on a machine with
# only one kind of device the other shape would never be executed by any test.
# It stays PRIVATE; nothing installed gains a header path from this.
target_include_directories(sushiblas_tests PRIVATE
    ${SB_TEST_COMMON_DIR}
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

# --- Link Libraries ---
target_link_libraries(sushiblas_tests PRIVATE
    sushiblas
    GTest::gtest
)

# --- Compile Policy ---
sushiblas_apply_build_policy(sushiblas_tests)

# One label per group; the "functional" umbrella is expressed by the consumer
# as a label regex (`ctest -L 'unit|integration|regression'`) rather than a
# second baked-in label, which does not survive gtest_discover_tests on
# current CMake (see sushiruntime/sushiengine tests/CMakeLists.txt).
gtest_discover_tests(sushiblas_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Unit_*"
    PROPERTIES LABELS "unit"
)
gtest_discover_tests(sushiblas_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Integration_*"
    PROPERTIES LABELS "integration"
)
gtest_discover_tests(sushiblas_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Regression_*"
    PROPERTIES LABELS "regression"
)

# --- Post-Build: Copy the runtime DLL ---
# Tests need the same DLL as the app to run on Windows. hwloc is a private
# implementation detail of SushiRuntime's own build and never reaches
# consumers (see examples/CMakeLists.txt).
if(WIN32)
    add_custom_command(TARGET sushiblas_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiblas_tests>
        COMMENT "Copying runtime DLL to test directory..."
    )
endif()
