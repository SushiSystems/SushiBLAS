# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

cmake_minimum_required(VERSION 3.20)

# An out-of-tree consumer of an INSTALLED SushiBLAS.
#
# This project is deliberately not part of the main build. It configures
# against the package the way a third party (SushiAI) would - find_package,
# nothing else, no access to sushiblas's source tree - because that is the
# only way to test the thing that actually breaks: a package that works only
# when the source tree happens to be next to it is not a package. Mirrors
# sushiruntime/tests/package/ exactly (see SushiRuntime's BB-4.6).
#
# Build and run it with:
#
#   sb build
#   sb test --suite package
#
# That installs to build/package/prefix, configures this project into
# build/package/consumer against it, builds it and runs it. It is not part of
# `sb test --suite all`, which covers only the CTest tests; `all` says so on
# every run rather than leaving the gap to be discovered.
#
# Two things that were easy to lose an hour to before the CLI handled them,
# both host-environment rather than package problems. They are recorded here
# because they explain what `sb test --suite package` is doing on your behalf,
# and what to look at if it ever regresses:
#
#   * SushiRuntime and SushiLog are packages the CLI built or reused, not part of
#     this build, so the install prefix holds SushiBLAS alone.
#   * On Windows the configure must run under the same MSVC environment `sb`
#     loads for its own builds (vcvars64: INCLUDE, LIB, and a resource compiler
#     clang-cl's CMake platform module insists on), and the built binary needs
#     sushiruntime.dll and hwloc-15.dll beside it or it exits 0xC0000135 before
#     main. In-tree targets get those from a POST_BUILD copy; the CLI stands in
#     for the consumer and deploys them (_deploy_consumer_dlls), which is what a
#     real consumer would have to do itself.
#
# The compiler must still be the SYCL one - not because SushiBLAS's kernels
# are header templates instantiated here (they are not: every kernel is
# compiled into the sushiblas static library already), but because the
# public headers include <sycl/sycl.hpp> and use sycl::event/sycl::queue in
# their own signatures, so this translation unit needs a SYCL-aware compiler
# to parse them at all. See docs/guides/INTEGRATION.md.

project(SushiBLASPackageConsumer LANGUAGES CXX)

find_package(SushiBLAS REQUIRED)

# --- Flag hygiene ------------------------------------------------------------
#
# The assertion SB-6's split exists to make: a consumer inherits nothing but
# SushiBLAS's include paths and library, plus whatever SushiRuntime itself
# legitimately exports (its floating-point contract). Warnings, -Werror,
# sanitizers and the ISA of the machine that built the package are
# development choices of this tree, and a consumer silently compiling under
# them gets build failures whose cause is invisible from its own source tree.
#
# Checked here rather than in sushiblas's own tests because "what a consumer
# inherits" is only observable from the consumer side - in-tree, every target
# links the policy targets directly and the leak would be invisible.
get_target_property(_sb_interface_options
    SushiBLAS::sushiblas INTERFACE_COMPILE_OPTIONS)
if(NOT _sb_interface_options)
    set(_sb_interface_options "")
endif()

get_target_property(_sb_interface_links
    SushiBLAS::sushiblas INTERFACE_LINK_LIBRARIES)
if(NOT _sb_interface_links)
    set(_sb_interface_links "")
endif()

set(_sb_forbidden_flags "-Werror" "-Wall" "-Wextra" "-pedantic"
                        "/W4" "/permissive-"
                        "-march=" "/arch:" "-fsanitize")

foreach(_sb_flag IN LISTS _sb_forbidden_flags)
    string(FIND "${_sb_interface_options}" "${_sb_flag}" _sb_found)
    if(NOT _sb_found EQUAL -1)
        message(FATAL_ERROR
            "SushiBLAS's exported interface leaks '${_sb_flag}' to consumers. "
            "Development flags belong on the PRIVATE policy targets; see "
            "cmake/SushiBLASBuildPolicy.cmake. Full interface: ${_sb_interface_options}")
    endif()
endforeach()

foreach(_sb_leaked IN ITEMS sushiblas_warnings sushiblas_perf sushiblas_sanitizers)
    string(FIND "${_sb_interface_links}" "${_sb_leaked}" _sb_found)
    if(NOT _sb_found EQUAL -1)
        message(FATAL_ERROR
            "SushiBLAS's exported interface names the in-tree policy target "
            "'${_sb_leaked}'. It should be wrapped in BUILD_INTERFACE. "
            "Full link interface: ${_sb_interface_links}")
    endif()
endforeach()

# The device options belong on SushiBLAS::device_code. On the sushiblas target they would put
# every unit of this consumer through the device passes.
if(NOT TARGET SushiBLAS::device_code OR NOT TARGET SushiBLAS::sycl_headers)
    message(FATAL_ERROR "The package exports no SushiBLAS::device_code or SushiBLAS::sycl_headers.")
endif()
get_target_property(_sb_device_options SushiBLAS::device_code INTERFACE_COMPILE_OPTIONS)
if(_sb_device_options AND "${_sb_interface_options}" MATCHES "-fsycl")
    message(FATAL_ERROR
        "SushiBLAS::sushiblas publishes a device compile option; it belongs on "
        "SushiBLAS::device_code. Full interface: ${_sb_interface_options}")
endif()

message(STATUS "SushiBLAS interface compile options: ${_sb_interface_options}")
message(STATUS "SushiBLAS interface link libraries:  ${_sb_interface_links}")

# --- The consumer ------------------------------------------------------------

add_executable(sushiblas_package_consumer main.cpp)

# main.cpp hands a functor to Dispatch::execute_binary, which instantiates a kernel, so its one
# unit compiles for the device.
target_link_libraries(sushiblas_package_consumer PRIVATE
    SushiBLAS::sushiblas
    SushiBLAS::device_code
)

# C++20 on purpose, and the one place in this repo that is. SushiBLAS itself is built at
# C++17 now; this consumer stays at 20 because that is what SushiAI is today, and because
# it is the only assertion that the standard change did not break the API. The public
# headers spell their spans `SushiRuntime::span`, which resolves to `std::span` here and
# to SushiRuntime's shim inside the library - and this target compiling proves the
# difference stays confined to inline code and never reaches a linked symbol.
set_target_properties(sushiblas_package_consumer PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
