# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the host matrices the sparse tests build from and the products they expect.
sushiblas_add_module(
    NAME sparse_fixtures
    LAYER foundation
    INTERNAL
    DEPENDS
        core
        arithmetic
    EXTERNAL
        GTest::gtest)
