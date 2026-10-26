# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The project umbrella, SushiBLAS/SushiBLAS.hpp, is this module's whole public surface. It
# includes every public header, so it depends on every installed module.
sushiblas_add_module(
    NAME umbrella
    LAYER library
    DEPENDS
        arithmetic
        core
        cost
        geometry
        graph_primitives
        launch
        orthonormal
        reduction
        reference
        sparse_assembly
        substitution
        sparse_storage
        tensor
        blas_seam_kernels
        engine
        functor_map
        operands
        blas
        elementwise
        experimental
        io
        logic
        nonlinear
        random
        reductions
        sparse
        spatial
        signal
        blend
        index)

# The tests that cross several operation families and so belong to no single family. The pack
# test needs the cluster module, which only SB_ENABLE_DISTRIBUTED builds.
set(sushiblas_umbrella_distributed_tests "")
if(SB_ENABLE_DISTRIBUTED)
    set(sushiblas_umbrella_distributed_tests test_SushiBLAS/test_pack_ordered_sum.cpp)
endif()

sushiblas_add_test_program(
    MODULE umbrella
    TESTS
        ${sushiblas_umbrella_distributed_tests}
    DEVICE_TESTS
        test_SushiBLAS/test_dependency_stress.cpp
        test_SushiBLAS/test_dispatch_complex.cpp
        test_SushiBLAS/test_dispatch_seam.cpp
        test_SushiBLAS/test_empty_tensors.cpp
        test_SushiBLAS/test_framework_sim.cpp
        test_SushiBLAS/test_integer_dtypes.cpp
        test_SushiBLAS/test_math_validation.cpp
        test_SushiBLAS/test_operand_lifetime.cpp
        test_SushiBLAS/test_regression.cpp
        test_SushiBLAS/test_spatial_random_io_validation.cpp
        test_SushiBLAS/test_storage_offset.cpp
        test_SushiBLAS/test_task_dependencies.cpp
    DEVICE_BENCHMARKS
        benchmark_elementwise.cpp)
