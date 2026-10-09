# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME support
    LAYER foundation
    DEPENDS
        core)

sushiblas_add_module_tests(
    MODULE support
    SOURCES
        test_complex_arithmetic.cpp
        test_launch_shapes.cpp
        test_reduce_plans.cpp
        test_reduce_segments.cpp
        test_reference_accumulator.cpp
        test_running_solution.cpp
        test_serial_loops.cpp
        test_sparse_assembly.cpp
        test_spatial_geometry.cpp)
