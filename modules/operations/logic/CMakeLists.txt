# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME logic
    LAYER operations
    SOURCES
        all.cpp
        any.cpp
        equal.cpp
        greater.cpp
        greater_equal.cpp
        less.cpp
        less_equal.cpp
        logical_and.cpp
        logical_not.cpp
        logical_or.cpp
        logical_xor.cpp
        not_equal.cpp
        where.cpp
    DEPENDS
        core
        tensor
        engine
        functor_map)

sushiblas_add_module_tests(
    MODULE logic
    SOURCES
        test_all.cpp
        test_any.cpp
        test_equal.cpp
        test_greater.cpp
        test_greater_equal.cpp
        test_less.cpp
        test_less_equal.cpp
        test_logical_and.cpp
        test_logical_not.cpp
        test_logical_or.cpp
        test_logical_xor.cpp
        test_not_equal.cpp
        test_where.cpp)
