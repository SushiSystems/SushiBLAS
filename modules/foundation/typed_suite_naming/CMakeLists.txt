# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Test-only: the name generator a typed suite passes to GoogleTest.
sushiblas_add_module(
    NAME typed_suite_naming
    LAYER foundation
    INTERNAL
    EXTERNAL
        GTest::gtest)

sushiblas_add_test_program(
    MODULE typed_suite_naming
    TESTS
        test_default_type_names.cpp)
