# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: SushiBLASTest, the fixture that gives each test a fresh Engine.
sushiblas_add_module(
    NAME engine_fixture
    LAYER engine
    INTERNAL
    DEPENDS
        core
        tensor
        engine
        test_runner
        tensor_expectations
    EXTERNAL
        GTest::gtest)
