# --- Find GTest ---
find_package(GTest CONFIG REQUIRED)

# --- Define Test Executable ---
add_executable(sushiblas_tests
    test_main.cpp

    # Level 3
    blas/level3/test_gemm.cpp
    
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
# Tests need the same DLLs as the app to run.
add_custom_command(TARGET sushiblas_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/sushiruntime.dll"
        "${CMAKE_SOURCE_DIR}/lib/hwloc-15.dll"
        $<TARGET_FILE_DIR:sushiblas_tests>
    COMMENT "Copying runtime DLLs to test directory..."
)
