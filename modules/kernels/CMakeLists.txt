# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The kernels tier: the reference and SYCL kernels, which know no tensor and no engine.
add_subdirectory(blas_seam_kernels)
add_subdirectory(blas_kernels)
add_subdirectory(random_kernels)
add_subdirectory(sparse_kernels)
add_subdirectory(spatial_kernels)

# Test-only modules: a test program includes them, the library never does.
if(SB_BUILD_TESTS)
    add_subdirectory(packed_product_harness)
endif()
