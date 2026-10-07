# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# ============================================================================
# SushiBLAS test pipeline
#
# One binary, sushiblas_tests, covering functional/{unit,integration,
# regression}. Tests are grouped into CTest labels (unit / integration /
# regression) from the GTest suite-name prefix convention (Unit_* /
# Integration_* / Regression_*), matching sushiruntime/sushiengine, so
# `ctest -L unit` (or the umbrella regex the CLI uses) selects a sub-suite and
# new tests are picked up automatically as long as they follow the convention.
#
# SB-3: elementwise/nonlinear/logic/reductions/random are enabled (pure SYCL,
# no oneMKL dependency). SB-3 follow-up: BLAS level1/level2/level3-trsm are
# also enabled now, backed by from-scratch reference+SYCL kernels (see
# kernels/{reference,sycl}/blas/{level1,level2,level3}/) instead of oneMKL,
# which was fully removed from this project's CMake when GEMM/SYRK were
# ported off it (see src/CMakeLists.txt). All BLAS levels now compile.
# ============================================================================

cmake_minimum_required(VERSION 3.22) # gtest_discover_tests(TEST_FILTER ...)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

set(SB_TEST_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_executable(sushiblas_tests
    common/test_main.cpp

    # --- Level 1 ---
    functional/unit/test_asum.cpp
    functional/unit/test_axpy.cpp
    functional/unit/test_copy.cpp
    functional/unit/test_dot.cpp
    functional/unit/test_iamax.cpp
    functional/unit/test_nrm2.cpp
    functional/unit/test_rot.cpp
    functional/unit/test_scal.cpp
    functional/unit/test_swap.cpp

    # --- Level 2 ---
    functional/unit/test_gemv.cpp
    functional/unit/test_ger.cpp
    functional/unit/test_symv.cpp
    functional/unit/test_syr.cpp
    functional/unit/test_syr2.cpp
    functional/unit/test_trmv.cpp
    functional/unit/test_trsv.cpp

    # --- Level 3 ---
    functional/unit/test_gemm.cpp
    functional/unit/test_syrk.cpp
    functional/unit/test_trsm.cpp

    # --- BLAS value types, bricks and the complex routines ---
    functional/unit/test_blas_types.cpp
    functional/unit/test_blas_scalar.cpp
    functional/unit/test_complex_arithmetic.cpp
    functional/unit/test_operand_access.cpp
    functional/unit/test_running_solution.cpp
    functional/unit/test_reference_accumulator.cpp
    functional/unit/test_blas_complex.cpp

    # --- Level 3 completion, band, packed, rotations and the packed GEMM path ---
    functional/unit/test_symmetric_access.cpp
    functional/unit/test_symm.cpp
    functional/unit/test_syr2k.cpp
    functional/unit/test_triangular_plan.cpp
    functional/unit/test_triangular_parallel.cpp
    functional/unit/test_givens.cpp
    functional/unit/test_rotg.cpp
    functional/unit/test_rotm.cpp
    functional/unit/test_rotmg.cpp
    functional/unit/test_band_access.cpp
    functional/unit/test_packed_access.cpp
    functional/unit/test_structured_operands.cpp
    functional/unit/test_blas_structured_validation.cpp
    functional/unit/test_blas_band.cpp
    functional/unit/test_blas_packed.cpp
    functional/unit/test_packed_product_plan.cpp
    functional/unit/test_packed_panel.cpp
    functional/unit/test_packed_micro_kernel.cpp
    functional/unit/test_packed_product.cpp
    functional/unit/test_packed_syrk.cpp
    functional/unit/test_packed_symm.cpp

    # --- Spatial (windowed movement over an N-D grid) ---
    functional/unit/test_spatial_geometry.cpp
    functional/unit/test_spatial_im2col.cpp
    functional/unit/test_spatial_col2im.cpp
    functional/unit/test_spatial_pool_max.cpp
    functional/unit/test_spatial_pool_avg.cpp
    functional/unit/test_spatial_upsample.cpp
    functional/unit/test_spatial_channels.cpp
    functional/unit/test_spatial_broadcast.cpp
    functional/unit/test_mean_spatial.cpp
    functional/unit/test_spatial_half.cpp
    functional/unit/test_spatial_window_dot.cpp
    functional/unit/test_channel_bias_epilogue.cpp

    # --- Nonlinear ---
    functional/unit/test_relu.cpp
    functional/unit/test_leaky_relu.cpp
    functional/unit/test_sigmoid.cpp
    functional/unit/test_tanh.cpp
    functional/unit/test_elu.cpp
    functional/unit/test_silu.cpp
    functional/unit/test_gelu.cpp
    functional/unit/test_softplus.cpp

    # --- Math: Random ---
    functional/unit/test_constant.cpp
    functional/unit/test_distributions.cpp
    functional/unit/test_initializers.cpp
    functional/unit/test_normal.cpp
    functional/unit/test_orthogonal.cpp
    functional/unit/test_shuffle.cpp
    functional/unit/test_uniform.cpp

    # --- Dispatch seam (the public SushiBLAS::Dispatch generic executors) ---
    functional/unit/test_dispatch_complex.cpp
    functional/unit/test_dispatch_seam.cpp
    functional/unit/test_read_binding.cpp
    functional/unit/test_functor_reference.cpp
    functional/unit/test_random_reference.cpp

    # --- Tensor views (every op family against a non-zero storage_offset) ---
    functional/unit/test_storage_offset.cpp
    functional/unit/test_tensor_view.cpp

    # --- Element types (which ops serve INT32/INT64, and which refuse them) ---
    functional/unit/test_integer_dtypes.cpp

    # --- Elementwise ---
    functional/unit/test_abs.cpp
    functional/unit/test_acos.cpp
    functional/unit/test_acosh.cpp
    functional/unit/test_add.cpp
    functional/unit/test_asin.cpp
    functional/unit/test_asinh.cpp
    functional/unit/test_atan.cpp
    functional/unit/test_atanh.cpp
    functional/unit/test_ceil.cpp
    functional/unit/test_clamp.cpp
    functional/unit/test_convert.cpp
    functional/unit/test_cos.cpp
    functional/unit/test_cosh.cpp
    functional/unit/test_div.cpp
    functional/unit/test_exp.cpp
    functional/unit/test_floor.cpp
    functional/unit/test_fmod.cpp
    functional/unit/test_log.cpp
    functional/unit/test_max.cpp
    functional/unit/test_min.cpp
    functional/unit/test_mul.cpp
    functional/unit/test_neg.cpp
    functional/unit/test_pow.cpp
    functional/unit/test_reciprocal.cpp
    functional/unit/test_remainder.cpp
    functional/unit/test_round.cpp
    functional/unit/test_sin.cpp
    functional/unit/test_sinh.cpp
    functional/unit/test_sqrt.cpp
    functional/unit/test_square.cpp
    functional/unit/test_sub.cpp
    functional/unit/test_tan.cpp

    # --- Reductions ---
    functional/unit/test_reductions.cpp
    functional/unit/test_reduction_determinism.cpp
    functional/unit/test_axis_reductions.cpp
    functional/unit/test_count_nonfinite.cpp
    functional/unit/test_reduce_plans.cpp
    functional/unit/test_reduce_segments.cpp

    # --- Logic ---
    functional/unit/test_all.cpp
    functional/unit/test_any.cpp
    functional/unit/test_equal.cpp
    functional/unit/test_greater.cpp
    functional/unit/test_greater_equal.cpp
    functional/unit/test_less.cpp
    functional/unit/test_less_equal.cpp
    functional/unit/test_logical_and.cpp
    functional/unit/test_logical_not.cpp
    functional/unit/test_logical_or.cpp
    functional/unit/test_logical_xor.cpp
    functional/unit/test_not_equal.cpp
    functional/unit/test_where.cpp

    # --- Operand validation and execution modes ---
    functional/unit/test_operand_check.cpp
    functional/unit/test_execution_mode.cpp
    functional/unit/test_blas_validation.cpp
    functional/unit/test_math_validation.cpp
    functional/unit/test_spatial_random_io_validation.cpp

    # --- Reference kernels and SYCL parity ---
    functional/unit/test_level1_reference.cpp
    functional/unit/test_level2_reference.cpp
    functional/unit/test_trsm_reference.cpp
    functional/unit/test_syrk_trsm_parity.cpp
    functional/unit/test_level1_parallel_reductions.cpp
    functional/unit/test_level2_sycl_parity.cpp

    # --- Empty tensors, integer arithmetic, host transfer, runtime shapes ---
    functional/unit/test_empty_tensors.cpp
    functional/unit/test_integer_arithmetic.cpp
    functional/unit/test_host_transfer.cpp
    functional/unit/test_runtime_shapes.cpp
    functional/unit/test_random_large_lambda.cpp
    functional/unit/test_sum_edge_cases.cpp
    functional/unit/test_max_min_nan.cpp
    functional/unit/test_complex_nonlinear.cpp
    functional/unit/test_blas_half_accumulation.cpp
    functional/unit/test_reference_mode_guard.cpp
    functional/unit/test_tensor_const_access.cpp
    functional/unit/test_rng_stream.cpp

    # --- Support layer and scratch ---
    functional/unit/test_scratch_shelf.cpp
    functional/unit/test_launch_shapes.cpp
    functional/unit/test_scratch_buffer.cpp
    functional/unit/test_serial_loops.cpp

    # --- Recorder, handles and the tensor factory ---
    functional/unit/test_task_handle.cpp
    functional/unit/test_task_recorder.cpp
    functional/unit/test_tensor_factory.cpp

    # --- Sparse ---
    functional/unit/test_sparse_types.cpp
    functional/unit/test_sparse_structure.cpp
    functional/unit/test_sparse_builder.cpp
    functional/unit/test_sparse_assembly.cpp
    functional/unit/test_sparse_validation.cpp
    functional/unit/test_sparse_level1.cpp
    functional/unit/test_sparse_spmv.cpp
    functional/unit/test_sparse_spsymv.cpp
    functional/unit/test_sparse_spmm.cpp
    functional/unit/test_sparse_sptrsv.cpp

    # --- Integration ---
    functional/integration/test_framework_sim.cpp
    functional/integration/test_dependency_stress.cpp

    # --- Regression ---
    functional/regression/test_regression.cpp
    functional/regression/test_spatial_determinism.cpp
    functional/regression/test_io.cpp
    functional/regression/test_logger.cpp
    functional/regression/test_operand_lifetime.cpp
    functional/regression/test_task_dependencies.cpp
    functional/regression/test_blas_symmetric_bits.cpp
    functional/regression/test_sparse_recording.cpp
)

# --- Include Directories ---
# The repo root is here for the same reason src/ has it: it is what makes
# `#include <kernels/...>` resolve. Tests need it to reach a kernel variant the
# public Engine API cannot select on its own - GEMM chooses between its
# local-tiled and no-local shapes from the device profile, so on a machine with
# only one kind of device the other shape would never be executed by any test.
# It stays PRIVATE; nothing installed gains a header path from this.
target_include_directories(sushiblas_tests PRIVATE
    ${SB_TEST_COMMON_DIR}
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
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
