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

    # --- Integration ---
    functional/integration/test_framework_sim.cpp
    functional/integration/test_dependency_stress.cpp

    # --- Regression ---
    functional/regression/test_regression.cpp
    functional/regression/test_spatial_determinism.cpp
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
