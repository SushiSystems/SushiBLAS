# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME logic
    LAYER operations
    DEVICE_SOURCES
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

sushiblas_add_test_program(
    MODULE logic
    DEVICE_TESTS
        test_dispatch.cpp
        test_logic/test_all.cpp
        test_logic/test_any.cpp
        test_logic/test_equal.cpp
        test_logic/test_greater.cpp
        test_logic/test_greater_equal.cpp
        test_logic/test_less.cpp
        test_logic/test_less_equal.cpp
        test_logic/test_logical_and.cpp
        test_logic/test_logical_not.cpp
        test_logic/test_logical_or.cpp
        test_logic/test_logical_xor.cpp
        test_logic/test_not_equal.cpp
        test_logic/test_where.cpp
    COMPILE_CHECKS
        test_logic_log_category.cpp)
