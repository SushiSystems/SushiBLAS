# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME index
    LAYER operations
    SOURCES
        index_ops.cpp
    DEPENDS
        core
        tensor
        engine
        functor_map)

sushiblas_add_test_program(
    MODULE index
    TESTS
        test_index_ops.cpp
        test_index_options.cpp)
