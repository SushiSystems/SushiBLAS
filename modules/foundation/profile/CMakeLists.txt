# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The profiler front: the SB_ZONE family of macros. Header-only. With SB_ENABLE_PROFILER off the
# macros expand to nothing and the module needs no SushiProfiler header.
set(profile_external "")
if(SB_ENABLE_PROFILER)
    set(profile_external SushiProfiler::SushiProfiler)
endif()

sushiblas_add_module(
    NAME profile
    LAYER foundation
    EXTERNAL
        ${profile_external})

# profile_config.hpp states SB_ENABLE_PROFILER and SB_PROFILE_LEVEL. It is generated once, under
# the include root <SushiBLAS/profile/profile_config.hpp> names, and that root is recorded with
# the installed ones, so the library puts it on its include path and the install ships it.
set(profile_generated "${CMAKE_BINARY_DIR}/generated")
configure_file(profile_config.hpp.in
    "${profile_generated}/SushiBLAS/profile/profile_config.hpp" @ONLY)
target_include_directories(sushiblas_profile INTERFACE "$<BUILD_INTERFACE:${profile_generated}>")
set_property(GLOBAL APPEND PROPERTY SUSHIBLAS_INSTALLED_INCLUDE_ROOTS "${profile_generated}")

sushiblas_add_test_program(
    MODULE profile
    TESTS
        profile_level_one.cpp
        profile_off.cpp
        test_profile.cpp
        test_profile_gate.cpp
        test_profile_keys.cpp)
