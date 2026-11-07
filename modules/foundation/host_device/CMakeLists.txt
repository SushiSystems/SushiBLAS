# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Test-only: the fixture that hands a kernel test the host CPU queue of the test environment,
# and the arrays it shares. The edge to test_runner is test-only, as both modules are.
sushiblas_add_module(
    NAME host_device
    LAYER foundation
    INTERNAL
    DEPENDS
        test_runner
    EXTERNAL
        GTest::gtest)

sushiblas_add_test_program(
    MODULE host_device
    TESTS
        test_host_device_test.cpp
        test_host_shared_allocator.cpp
        test_usm_shared_allocator.cpp
    DEVICE_TESTS
        test_shared_array.cpp
    COMPILE_CHECKS
        compile_host_device_test.cpp
        compile_shared_allocator.cpp)
