# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: serial host reductions a test takes its expected values from.
sushiblas_add_module(
    NAME serial_reductions
    LAYER foundation
    INTERNAL)

sushiblas_add_test_program(
    MODULE serial_reductions
    TESTS
        test_serial_reductions.cpp)
