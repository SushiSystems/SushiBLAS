# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME graph_primitives
    LAYER foundation)

sushiblas_add_test_program(
    MODULE graph_primitives
    TESTS
        test_combiners.cpp
        test_counting_sort.cpp
        test_element_window.cpp
        test_fixed_order_fold.cpp
        test_fixed_order_reduce.cpp
        test_fixed_order_scan.cpp
        test_invalid_graph.cpp
        test_prefix_scan.cpp
        test_radix_plan.cpp
        test_reduce/test_device_sized_reduce.cpp
        test_reduce/test_reduce.cpp
        test_segmented_reduce.cpp)
