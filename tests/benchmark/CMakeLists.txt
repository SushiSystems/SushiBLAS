# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# One executable per benchmark source. Each includes benchmark_common.hpp from this folder.
set(SB_BENCHMARKS benchmark_gemm benchmark_gemm_blocking benchmark_level1 benchmark_reductions benchmark_elementwise
    benchmark_band_packed benchmark_sparse benchmark_triangular)

foreach(benchmark IN LISTS SB_BENCHMARKS)
    add_executable(${benchmark} ${benchmark}.cpp)
    target_include_directories(${benchmark} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${benchmark} PRIVATE sushiblas)
    sushiblas_apply_build_policy(${benchmark})
    set_target_properties(${benchmark} PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/src")

    if(WIN32)
        add_custom_command(TARGET ${benchmark} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${benchmark}>
            COMMENT "Copying runtime DLL to benchmark directory..."
        )
    endif()
endforeach()

# benchmark_triangular times the TRSM kernel at block widths the public API does not select.
target_include_directories(benchmark_triangular PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
