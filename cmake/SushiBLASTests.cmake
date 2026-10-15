# SushiBLASTests.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASTests.cmake — the one shape of a test program.
#
# Single responsibility: build a GoogleTest program against the library and register its
# tests under the three CTest labels. A module's tests and the cross-module tests under
# tests/ go through the same function, so every program is linked, labelled and placed
# the same way.
#
# A test belongs to a label by the prefix of its GoogleTest suite name (Unit_*,
# Integration_*, Regression_*), as in SushiRuntime and SushiEngine. The "functional"
# umbrella is a label regex the CLI passes to ctest, not a fourth label.

# Every test program lands in one folder, so the runtime DLL is copied to one place and
# `sb run <program>` finds each of them where it found the single program before.
set(SB_TEST_OUTPUT_DIR "${PROJECT_BINARY_DIR}/tests")
# The main() every test program shares. It belongs to the test_runner module and is
# compiled into each program, never into the library.
set(SB_TEST_MAIN_SOURCE "${PROJECT_SOURCE_DIR}/modules/foundation/test_runner/source/test_main.cpp")

# Builds one test program from SOURCES, given relative to the calling folder.
#
#   sushiblas_add_test_program(NAME <target> SOURCES <file>...)
#
# The program links the library and the interface target that carries the INTERNAL
# modules' include roots: a test may include a kernel a consumer cannot, which is how a
# kernel variant the device profile would not pick on this machine still gets run.
function(sushiblas_add_test_program)
    cmake_parse_arguments(ARG "" "NAME" "SOURCES" ${ARGN})
    if("${ARG_NAME}" STREQUAL "" OR NOT ARG_SOURCES)
        message(FATAL_ERROR "sushiblas_add_test_program(): NAME and SOURCES are required.")
    endif()
    add_executable(${ARG_NAME}
        "${SB_TEST_MAIN_SOURCE}"
        ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME} PRIVATE
        sushiblas
        sushiblas_internal_headers
        GTest::gtest)
    sushiblas_apply_build_policy(${ARG_NAME})
    set_target_properties(${ARG_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${SB_TEST_OUTPUT_DIR}")
    foreach(label IN ITEMS unit integration regression)
        string(SUBSTRING "${label}" 0 1 initial)
        string(SUBSTRING "${label}" 1 -1 remainder)
        string(TOUPPER "${initial}" initial)
        gtest_discover_tests(${ARG_NAME}
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}"
            TEST_FILTER "${initial}${remainder}_*"
            PROPERTIES LABELS "${label}")
    endforeach()
    # The tests need the runtime DLL beside them on Windows. hwloc is private to
    # SushiRuntime's own build and never reaches a consumer.
    if(WIN32)
        add_custom_command(TARGET ${ARG_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${ARG_NAME}>
            COMMENT "Copying runtime DLL to test directory...")
    endif()
endfunction()

# Builds the test program of one module from the files under its tests/ folder.
#
#   sushiblas_add_module_tests(MODULE <module> SOURCES <file under tests/>...)
#
# Does nothing when SB_BUILD_TESTS is off.
function(sushiblas_add_module_tests)
    cmake_parse_arguments(ARG "" "MODULE" "SOURCES" ${ARGN})
    if(NOT SB_BUILD_TESTS)
        return()
    endif()
    set(test_sources "")
    foreach(source IN LISTS ARG_SOURCES)
        list(APPEND test_sources "tests/${source}")
    endforeach()
    sushiblas_add_test_program(NAME "sushiblas_${ARG_MODULE}_tests" SOURCES ${test_sources})
endfunction()
