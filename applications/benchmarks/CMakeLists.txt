# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The scenario benchmarks. Added only with SB_ENABLE_PROFILER (CMakeLists.txt): they use the
# profiling module. The program is built with SB_BUILD_BENCHMARKS, the tests of its parts with
# SB_BUILD_TESTS.

if(SB_BUILD_BENCHMARKS)
    add_executable(device_ceilings device_ceilings/main.cpp)
    sushiblas_configure_test_target(device_ceilings)
    sushiblas_apply_sycl(device_ceilings DEVICE_SOURCES device_ceilings/main.cpp)
    target_include_directories(device_ceilings PRIVATE device_ceilings)
    set_property(TARGET device_ceilings PROPERTY LABELS benchmark)
endif()

sushiblas_add_application_test_program(
    NAME device_ceilings
    DIRECTORY benchmarks/device_ceilings
    TESTS
        test_ceilings_options.cpp
        test_ceilings_statistics.cpp
        test_span_observer.cpp
    DEVICE_TESTS
        test_device_meter.cpp)
