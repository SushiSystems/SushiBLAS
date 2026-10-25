# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

cmake_minimum_required(VERSION 3.20)

# SushiBLAS — a head; SushiRuntime is the plugged-in component (the battery).
# Mirrors SushiEngine's own root CMakeLists.txt ordering exactly (both are
# head modules over the same backbone): Vcpkg + Toolchain run BEFORE
# project() to shape the toolchain file and pick the default SYCL compiler
# ahead of project()'s probe; everything else (the runtime dependency, the
# SYCL target policy) runs AFTER. See cmake/ for each stage.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Pre-project: toolchain shaping -------------------------------------------
include(Vcpkg)      # (Windows) locate vcpkg + set the toolchain file
include(Toolchain)  # pick the default SYCL compiler (clang++) before the probe

project(SushiBLAS VERSION 0.1.0 LANGUAGES CXX C)

# --- Post-project: runtime dependency (D3 layered resolution, SB-0), SYCL policy
option(SB_ENABLE_DISTRIBUTED "Build the cluster module over SushiRuntime's distributed layer" OFF)
if(SB_ENABLE_DISTRIBUTED)
    set(SR_ENABLE_DISTRIBUTED ON CACHE BOOL "Build SushiRuntime's distributed layer" FORCE)
endif()
include(SushiBLASSushiLog)  # SushiLog::SushiLog, before the runtime links it
include(Runtime)     # add_subdirectory(sushiruntime) -> SushiRuntime::SushiRuntime,
                      # add_sycl_to_target(), SR_SYCL_TOOLCHAIN
include(SyclTarget)  # the SYCL policy: sushiblas_apply_sycl(), sushiblas_publish_sycl()

# Compile policy. BuildPolicy composes the warning, architecture, and
# sanitizer targets and exposes sushiblas_apply_build_policy(<target>), which
# every target in the tree calls. A function rather than a directory-scope
# add_compile_options() so the in-tree-only flags never leak into the
# installed export set (see cmake/SushiBLASBuildPolicy.cmake). Removed here: the old
# global add_compile_options(-march=native / /arch:AVX2 / /fp:fast) block —
# baking the build machine's ISA and fast-math into every target
# unconditionally is exactly the anti-pattern SB-6 packaging exists to
# close. -fp:fast in particular silently reassociates and contracts FP ops,
# which would have fought SushiRuntime's own deterministic-FP contract that
# sushiblas already inherits transitively.
include(SushiBLASBuildPolicy)

# SB_ENABLE_LOGGING -> sushiblas_apply_logging(<target>). Separate from BuildPolicy because
# it is the one compile definition that must be PUBLIC, so a consumer compiles the same log
# calls the library does.
include(SushiBLASLogging)
include(SushiBLASModule)  # sushiblas_add_module() and the tier table

# --- Policies ---
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# --- Configuration ---
#
# C++17, matching SushiRuntime. One standard across the stack is not tidiness:
# SushiRuntime's own headers branch on __cplusplus - include/SushiRuntime/core/span.hpp
# is std::span at C++20 and its own shim below - and that branch reaches the compiled
# ABI (distributed/i_transport.hpp declares a virtual taking a SushiRuntime::span, whose
# mangled name differs between the two readings). SushiBLAS compiling at 20 against a
# SushiRuntime compiled at 17 put the two halves of one build on opposite sides of that
# branch. They are on the same side now, and SushiBLAS's own public API is unaffected
# because SushiRuntime::span *is* std::span for any consumer still on C++20.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Install layout & package location (SB-6) ---
include(GNUInstallDirs)
set(SB_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SushiBLAS"
    CACHE STRING "Where the SushiBLAS CMake package files are installed")

# The version literals in modules/foundation/core/include/SushiBLAS/core/version.hpp are hand-written, so
# they are greppable and need no generated header on the include path. That
# trade is only safe if they cannot silently disagree with the version the
# build and the package advertise — so the disagreement is a configure-time
# error, here, rather than a consumer discovering it later. Mirrors
# sushiruntime's own version-drift check exactly.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/modules/foundation/core/include/SushiBLAS/core/version.hpp"
     SB_VERSION_STRING_LINE
     REGEX "^#define SUSHIBLAS_VERSION_STRING ")
string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _sb_unused "${SB_VERSION_STRING_LINE}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR
        "Version mismatch: project() says ${PROJECT_VERSION}, "
        "modules/foundation/core/include/SushiBLAS/core/version.hpp says ${CMAKE_MATCH_1}. "
        "Update both — they are the same fact stated twice.")
endif()

# --- Subdirectories ---
# Examples and tests are built when SushiBLAS is the top-level project, not when a consumer
# pulls it in with add_subdirectory.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SB_IS_TOP_LEVEL ON)
else()
    set(SB_IS_TOP_LEVEL OFF)
endif()

# --- Testing switch ---
# Declared before the modules: a module registers its own test program, so the switch,
# GoogleTest and enable_testing() have to exist by the time the first module is read.
option(SB_BUILD_TESTS "Build and register the test program of each module from its mirror under tests/" ${SB_IS_TOP_LEVEL})
if(SB_BUILD_TESTS)
    if(CMAKE_VERSION VERSION_LESS 3.22)
        message(FATAL_ERROR "SB_BUILD_TESTS needs CMake 3.22 for gtest_discover_tests(TEST_FILTER).")
    endif()
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)
endif()
# Declared before the modules for the same reason: a module adds its benchmark programs, and
# the operations tier adds benchmark_harness, only when the switch is on.
option(SB_BUILD_BENCHMARKS "Build the benchmark programs of the modules under tests/" OFF)
include(SushiBLASTests)  # sushiblas_add_test_program(), sushiblas_add_application_test_program()

# --- Modules ---
# Every tier in order; the last one composes the `sushiblas` library from the object
# libraries and the include roots the others record.
add_subdirectory(modules)

option(SB_BUILD_EXAMPLES "Build the example programs under applications/examples/" ${SB_IS_TOP_LEVEL})
if(SB_BUILD_EXAMPLES)
    add_subdirectory(applications/examples)
endif()


# --- Package files (SB-6) ---
# Written after add_subdirectory(modules) so the sushiblas/SushiRuntime::SushiRuntime
# targets it references already exist.
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SushiBLASConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfig.cmake"
    INSTALL_DESTINATION "${SB_INSTALL_CMAKEDIR}"
)

# SameMinorVersion: the project is pre-1.0 and its API is not frozen, so a
# consumer asking for 0.2 must not be handed 0.3. Revisit at the v1.0.0
# freeze, where SameMajorVersion becomes the honest compatibility statement.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

# SushiBLASSyclHeaders.cmake goes with them: the Config file calls it to find the SYCL headers
# beside the consumer's compiler.
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SushiBLASConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SushiBLASSyclHeaders.cmake"
    DESTINATION "${SB_INSTALL_CMAKEDIR}"
)
