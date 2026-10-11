# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME graph_primitives
    LAYER foundation)

sushiblas_add_module_tests(
    MODULE graph_primitives
    SOURCES
        test_combiners.cpp
        test_counting_sort.cpp
        test_device_sized_primitives.cpp
        test_fixed_order_fold.cpp
        test_fixed_order_levels.cpp
        test_prefix_scan.cpp
        test_radix_plan.cpp
        test_reduce.cpp
        test_segmented_reduce.cpp)
