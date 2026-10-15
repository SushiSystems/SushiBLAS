# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the execution modes a routine's tests run it in.
sushiblas_add_module(
    NAME execution_mode_sweep
    LAYER engine
    INTERNAL
    DEPENDS
        engine)

sushiblas_add_test_program(
    MODULE execution_mode_sweep
    SIZE medium
    TESTS
        test_execution_mode_sweep.cpp)
