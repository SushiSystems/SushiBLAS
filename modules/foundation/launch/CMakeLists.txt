# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

sushiblas_add_module(
    NAME launch
    LAYER foundation
    DEPENDS
        core)

sushiblas_add_test_program(
    MODULE launch
    TESTS
        test_launch_1d.cpp)
