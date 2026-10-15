# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the name generator a typed suite passes to GoogleTest.
sushiblas_add_module(
    NAME typed_suite_naming
    LAYER foundation
    INTERNAL
    EXTERNAL
        GTest::gtest)
