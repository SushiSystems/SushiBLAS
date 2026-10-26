# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The operations tier: one module per operation family.
add_subdirectory(blas)
add_subdirectory(elementwise)
add_subdirectory(index)
add_subdirectory(nonlinear)
add_subdirectory(reductions)
add_subdirectory(random)
add_subdirectory(logic)
add_subdirectory(sparse)
add_subdirectory(spatial)
add_subdirectory(signal)
add_subdirectory(blend)
add_subdirectory(io)
add_subdirectory(experimental)

if(SB_BUILD_BENCHMARKS AND NOT SB_ENABLE_PROFILER)
    message(FATAL_ERROR
        "SB_BUILD_BENCHMARKS needs SB_ENABLE_PROFILER: the benchmark programs record "
        "through the profiling module.")
endif()

if(SB_BUILD_BENCHMARKS)
    add_subdirectory(benchmark_harness)
endif()
