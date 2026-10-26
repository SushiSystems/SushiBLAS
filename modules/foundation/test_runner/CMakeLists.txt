# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the GoogleTest environment that owns the runtime, and the main() of every
# test program. source/test_main.cpp is not a module source: sushiblas_add_test_program()
# compiles it into each test program (cmake/SushiBLASTests.cmake), so it never reaches
# the library.
sushiblas_add_module(
    NAME test_runner
    LAYER foundation
    INTERNAL
    DEPENDS
        core
    EXTERNAL
        GTest::gtest)

sushiblas_add_test_program(
    MODULE test_runner
    TESTS
        test_log_capture.cpp
        test_run_observer_capture.cpp
        test_test_environment.cpp
        test_test_runner_log_category.cpp)
