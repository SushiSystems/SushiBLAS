# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: the queue on the host CPU device a kernel test runs on, and the arrays it shares.
sushiblas_add_module(
    NAME host_device
    LAYER foundation
    INTERNAL
    EXTERNAL
        GTest::gtest)

sushiblas_add_test_program(
    MODULE host_device
    TESTS
        test_host_device_test.cpp
        test_host_queue.cpp
    DEVICE_TESTS
        test_shared_array.cpp)
