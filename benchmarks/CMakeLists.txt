# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# One executable per benchmark source. Each includes bench_common.hpp from this folder.
set(SB_BENCHMARKS bench_gemm bench_level1 bench_reductions bench_elementwise)

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
