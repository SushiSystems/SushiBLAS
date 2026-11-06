# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The kernels tier: the reference and SYCL kernels, which know no tensor and no engine.
add_subdirectory(blas_seam_kernels)
add_subdirectory(blas_kernels)
add_subdirectory(random_kernels)
add_subdirectory(sparse_kernels)
add_subdirectory(spatial_kernels)
add_subdirectory(signal_kernels)
add_subdirectory(linalg_kernels)
add_subdirectory(blend_kernels)

# Test-only modules: a test program includes them, the library never does.
if(SB_BUILD_TESTS)
    add_subdirectory(packed_product_harness)
endif()
