# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Test-only: SushiBLASTest, the fixture that gives each test a fresh Engine.
sushiblas_add_module(
    NAME engine_fixture
    LAYER engine
    INTERNAL
    DEPENDS
        core
        tensor
        engine
        recording_graph
        test_runner
        tensor_expectations
    EXTERNAL
        GTest::gtest)

sushiblas_add_test_program(
    MODULE engine_fixture
    TESTS
        test_host_engine_test.cpp
        test_host_recording_fixture.cpp
        test_recording_fixture.cpp
        test_sushiblas_test.cpp)
