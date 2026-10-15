# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The data tier: the containers an operation reads and writes.
add_subdirectory(tensor)
add_subdirectory(sparse_storage)

# Test-only modules: a test program includes them, the library never does.
if(SB_BUILD_TESTS)
    add_subdirectory(tensor_expectations)
endif()
