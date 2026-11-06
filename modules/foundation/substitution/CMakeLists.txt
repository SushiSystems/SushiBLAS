# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

sushiblas_add_module(
    NAME substitution
    LAYER foundation
    DEPENDS
        arithmetic)

sushiblas_add_test_program(
    MODULE substitution
    TESTS
        test_running_solution.cpp
    COMPILE_CHECKS
        compile_running_solution.cpp)
