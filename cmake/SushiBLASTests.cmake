# SushiBLASTests.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASTests.cmake — the one shape of a module's test program.
#
# Single responsibility: build the test program and the benchmark programs of one module from
# its mirror under tests/<tier>/<module>/, and register them under their labels. Every module
# goes through the same function, so every program is linked, labelled and placed the same way.
#
# A test belongs to a label by the prefix of its GoogleTest suite name (Unit_*,
# Integration_*, Regression_*, Device_*), as in SushiRuntime and SushiEngine. A benchmark
# program carries the target label `benchmark` and is never registered with CTest: it
# measures, it does not pass or fail. The "functional" umbrella is a label regex the CLI
# passes to ctest, not a label of its own.

# Every program lands in one folder, so the runtime DLL is copied to one place and
# `sb run <program>` finds each of them there.
set(SB_TEST_OUTPUT_DIR "${PROJECT_BINARY_DIR}/tests")
# The main() every test program shares. It belongs to the test_runner module and is
# compiled into each program, never into the library.
set(SB_TEST_MAIN_SOURCE "${PROJECT_SOURCE_DIR}/modules/foundation/test_runner/source/test_main.cpp")
# The labels a test program registers, each matched to the suite prefix of the same name.
set(SB_TEST_LABELS unit integration regression device)

# Links a program to the library and the INTERNAL modules' include roots, applies the build
# policy, places it in SB_TEST_OUTPUT_DIR and copies the runtime DLL beside it on Windows.
# A test may include a kernel a consumer cannot, which is how a kernel variant the device
# profile would not pick on this machine still gets run.
function(sushiblas_configure_test_target target)
    target_link_libraries(${target} PRIVATE
        sushiblas
        sushiblas_internal_headers)
    sushiblas_apply_build_policy(${target})
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${SB_TEST_OUTPUT_DIR}")
    # hwloc is private to SushiRuntime's own build and never reaches a consumer.
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying runtime DLL to test directory...")
    endif()
endfunction()

# Builds one GoogleTest program from the shared main() and the sources given, and registers
# its tests under the label each suite prefix names.
function(sushiblas_add_labelled_test_program target)
    add_executable(${target} "${SB_TEST_MAIN_SOURCE}" ${ARGN})
    target_link_libraries(${target} PRIVATE GTest::gtest)
    sushiblas_configure_test_target(${target})
    foreach(label IN LISTS SB_TEST_LABELS)
        string(SUBSTRING "${label}" 0 1 initial)
        string(SUBSTRING "${label}" 1 -1 remainder)
        string(TOUPPER "${initial}" initial)
        gtest_discover_tests(${target}
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}"
            TEST_FILTER "${initial}${remainder}_*"
            PROPERTIES LABELS "${label}")
    endforeach()
endfunction()

# Builds the test program and the benchmark programs of one module.
#
#   sushiblas_add_test_program(MODULE <module>
#       [TESTS <file>...]
#       [BENCHMARKS <file>...])
#
# Each file is given relative to tests/<tier>/<module>/, the module's mirror. TESTS become
# the program `sushiblas_<module>_tests` when SB_BUILD_TESTS is on; each of BENCHMARKS becomes
# a program named after its file when SB_BUILD_BENCHMARKS is on.
function(sushiblas_add_test_program)
    cmake_parse_arguments(ARG "" "MODULE" "TESTS;BENCHMARKS" ${ARGN})
    if("${ARG_MODULE}" STREQUAL "" OR (NOT ARG_TESTS AND NOT ARG_BENCHMARKS))
        message(FATAL_ERROR "sushiblas_add_test_program(): MODULE and TESTS or BENCHMARKS are required.")
    endif()
    sushiblas_module_layer("${ARG_MODULE}" layer)
    if("${layer}" STREQUAL "")
        message(FATAL_ERROR "sushiblas_add_test_program(): ${ARG_MODULE} has no tier in SUSHIBLAS_MODULE_LAYERS.")
    endif()
    set(mirror "${PROJECT_SOURCE_DIR}/tests/${layer}/${ARG_MODULE}")

    if(SB_BUILD_TESTS AND ARG_TESTS)
        set(target "sushiblas_${ARG_MODULE}_tests")
        list(TRANSFORM ARG_TESTS PREPEND "${mirror}/" OUTPUT_VARIABLE test_sources)
        sushiblas_add_labelled_test_program(${target} ${test_sources})
    endif()

    if(SB_BUILD_BENCHMARKS)
        foreach(source IN LISTS ARG_BENCHMARKS)
            get_filename_component(benchmark "${source}" NAME_WE)
            add_executable(${benchmark} "${mirror}/${source}")
            sushiblas_configure_test_target(${benchmark})
            set_property(TARGET ${benchmark} PROPERTY LABELS benchmark)
        endforeach()
    endif()
endfunction()

# Builds the test program of one application from its mirror under tests/applications/.
#
#   sushiblas_add_application_test_program(NAME <name> DIRECTORY <application folder>
#       SOURCES <application source>...
#       TESTS <file>...)
#
# DIRECTORY is relative to applications/; SOURCES are relative to it and TESTS to its mirror
# tests/applications/<DIRECTORY>/. The program is `sushiblas_<name>_tests`, and it sees the
# application's own headers as the application does.
function(sushiblas_add_application_test_program)
    cmake_parse_arguments(ARG "" "NAME;DIRECTORY" "SOURCES;TESTS" ${ARGN})
    if("${ARG_NAME}" STREQUAL "" OR "${ARG_DIRECTORY}" STREQUAL "" OR NOT ARG_TESTS)
        message(FATAL_ERROR "sushiblas_add_application_test_program(): NAME, DIRECTORY and TESTS are required.")
    endif()
    if(NOT SB_BUILD_TESTS)
        return()
    endif()
    set(application "${PROJECT_SOURCE_DIR}/applications/${ARG_DIRECTORY}")
    set(mirror "${PROJECT_SOURCE_DIR}/tests/applications/${ARG_DIRECTORY}")
    list(TRANSFORM ARG_SOURCES PREPEND "${application}/" OUTPUT_VARIABLE application_sources)
    list(TRANSFORM ARG_TESTS PREPEND "${mirror}/" OUTPUT_VARIABLE test_sources)
    set(target "sushiblas_${ARG_NAME}_tests")
    sushiblas_add_labelled_test_program(${target} ${application_sources} ${test_sources})
    target_include_directories(${target} PRIVATE "${application}")
endfunction()
