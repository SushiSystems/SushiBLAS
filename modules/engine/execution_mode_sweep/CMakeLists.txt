# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Test-only: the execution modes a routine's tests run it in.
sushiblas_add_module(
    NAME execution_mode_sweep
    LAYER engine
    INTERNAL
    DEPENDS
        engine)

sushiblas_add_test_program(
    MODULE execution_mode_sweep
    COMPILE_CHECKS
        compile_execution_mode_sweep.cpp)
