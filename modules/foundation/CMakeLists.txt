# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The foundation tier: what every other module is written in.
add_subdirectory(core)
add_subdirectory(arithmetic)
add_subdirectory(launch)
add_subdirectory(reduction)
add_subdirectory(reference)
add_subdirectory(substitution)
add_subdirectory(orthonormal)
add_subdirectory(sparse_assembly)
add_subdirectory(geometry)
add_subdirectory(graph_primitives)

# Test-only modules: a test program includes them, the library never does.
if(SB_BUILD_TESTS)
    add_subdirectory(blas_test_oracle)
    add_subdirectory(host_device)
    add_subdirectory(small_integers)
    add_subdirectory(serial_reductions)
    add_subdirectory(test_runner)
    add_subdirectory(sparse_fixtures)
    add_subdirectory(typed_suite_naming)
endif()
