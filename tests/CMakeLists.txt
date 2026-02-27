# --- Find GTest ---
find_package(GTest CONFIG REQUIRED)

# --- Define Test Executable ---
add_executable(sushiblas_tests
    test_main.cpp

    # Level 1
    blas/level1/test_asum.cpp
    blas/level1/test_axpy.cpp
    blas/level1/test_copy.cpp
    blas/level1/test_dot.cpp
    blas/level1/test_iamax.cpp
    blas/level1/test_nrm2.cpp
    blas/level1/test_rot.cpp
    blas/level1/test_scal.cpp
    blas/level1/test_swap.cpp

    # Level 2
    blas/level2/test_gemv.cpp
    blas/level2/test_ger.cpp
    blas/level2/test_symv.cpp
    blas/level2/test_syr.cpp
    blas/level2/test_syr2.cpp
    blas/level2/test_trmv.cpp
    blas/level2/test_trsv.cpp

    # Level 3
    blas/level3/test_gemm.cpp
    blas/level3/test_trsm.cpp
    blas/level3/test_syrk.cpp
    
    # Nonlinear
    math/nonlinear/test_relu.cpp
    math/nonlinear/test_leaky_relu.cpp
    math/nonlinear/test_sigmoid.cpp
    math/nonlinear/test_tanh.cpp
    math/nonlinear/test_elu.cpp
    math/nonlinear/test_silu.cpp
    math/nonlinear/test_gelu.cpp
    math/nonlinear/test_softplus.cpp
    
    # Math: Random
    math/random/test_constant.cpp
    math/random/test_distributions.cpp
    math/random/test_initializers.cpp
    math/random/test_normal.cpp
    math/random/test_uniform.cpp
    
    # Elementwise
    math/elementwise/test_abs.cpp
    math/elementwise/test_acos.cpp
    math/elementwise/test_acosh.cpp
    math/elementwise/test_add.cpp
    math/elementwise/test_asin.cpp
    math/elementwise/test_asinh.cpp
    math/elementwise/test_atan.cpp
    math/elementwise/test_atanh.cpp
    math/elementwise/test_ceil.cpp
    math/elementwise/test_clamp.cpp
    math/elementwise/test_cos.cpp
    math/elementwise/test_cosh.cpp
    math/elementwise/test_div.cpp
    math/elementwise/test_exp.cpp
    math/elementwise/test_floor.cpp
    math/elementwise/test_fmod.cpp
    math/elementwise/test_log.cpp
    math/elementwise/test_max.cpp
    math/elementwise/test_min.cpp
    math/elementwise/test_mul.cpp
    math/elementwise/test_neg.cpp
    math/elementwise/test_pow.cpp
    math/elementwise/test_reciprocal.cpp
    math/elementwise/test_remainder.cpp
    math/elementwise/test_round.cpp
    math/elementwise/test_sin.cpp
    math/elementwise/test_sinh.cpp
    math/elementwise/test_sqrt.cpp
    math/elementwise/test_square.cpp
    math/elementwise/test_sub.cpp
    math/elementwise/test_tan.cpp
    
    # Logic
    logic/test_all.cpp
    logic/test_any.cpp
    logic/test_equal.cpp
    logic/test_greater.cpp
    logic/test_greater_equal.cpp
    logic/test_less.cpp
    logic/test_less_equal.cpp
    logic/test_logical_and.cpp
    logic/test_logical_not.cpp
    logic/test_logical_or.cpp
    logic/test_logical_xor.cpp
    logic/test_not_equal.cpp
    logic/test_where.cpp
    
    # Framework Simulation
    runtime/test_framework_sim.cpp
    
    # Stress tests
    runtime/test_dependency_stress.cpp
)

# --- Include Directories ---
target_include_directories(sushiblas_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
)

# --- Link Libraries ---
target_link_libraries(sushiblas_tests PRIVATE
    sushiblas
    GTest::gtest
)

# --- Test Registration ---
add_test(NAME AllTests COMMAND sushiblas_tests)

# --- Post-Build: Copy DLLs ---
# Tests need the same DLLs as the app to run on Windows.
if(WIN32)
    add_custom_command(TARGET sushiblas_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/lib/sushiruntime.dll"
            "${CMAKE_SOURCE_DIR}/lib/hwloc-15.dll"
            $<TARGET_FILE_DIR:sushiblas_tests>
        COMMENT "Copying runtime DLLs to test directory..."
    )
endif()
