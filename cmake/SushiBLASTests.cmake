# SushiBLASTests.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASTests.cmake — the one shape of a module's test program.
#
# Single responsibility: build the test program and the benchmark programs of one module from
# its mirror under tests/<tier>/<module>/, and register them under their size. Every module
# goes through the same function, so every program is linked, labelled and placed the same way.
#
# Every test carries exactly one size label, small, medium or large, and the timeout of that
# size. The size is declared per program. A small program runs without the shared runtime
# environment, so it starts no worker thread and opens no device. A test listed as quarantined
# also carries the label `quarantined`, which takes it out of every gating stage. A benchmark
# program carries the target label `benchmark` and is never registered with CTest: it measures,
# it does not pass or fail.

# Every program lands in one folder, so the runtime DLL is copied to one place and
# `sb run <program>` finds each of them there.
set(SB_TEST_OUTPUT_DIR "${PROJECT_BINARY_DIR}/tests")
# The main() of a medium or large program, which owns the shared runtime environment, and of a
# small program, which owns none. Both belong to the test_runner module and are compiled into
# each program, never into the library.
set(SB_TEST_MAIN_SOURCE "${PROJECT_SOURCE_DIR}/modules/foundation/test_runner/source/test_main.cpp")
set(SB_SMALL_TEST_MAIN_SOURCE
    "${PROJECT_SOURCE_DIR}/modules/foundation/test_runner/source/small_test_main.cpp")
# The timeout in seconds of each size.
set(SB_TEST_TIMEOUT_small 60)
set(SB_TEST_TIMEOUT_medium 300)
set(SB_TEST_TIMEOUT_large 900)

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

# Builds one GoogleTest program from the main() of its size and the sources given, and registers
# its tests under that size and its timeout. Each name in QUARANTINED, a GoogleTest
# `Suite.Case` name, is registered with the label `quarantined` as well.
#
#   sushiblas_add_sized_test_program(<target> SIZE <small|medium|large>
#       [QUARANTINED <Suite.Case>...]
#       SOURCES <file>...)
function(sushiblas_add_sized_test_program target)
    cmake_parse_arguments(ARG "" "SIZE" "QUARANTINED;SOURCES" ${ARGN})
    if(NOT ARG_SIZE MATCHES "^(small|medium|large)$")
        message(FATAL_ERROR
            "${target}: SIZE is '${ARG_SIZE}', expected small, medium or large.")
    endif()
    if(ARG_SIZE STREQUAL "small")
        set(main_source "${SB_SMALL_TEST_MAIN_SOURCE}")
    else()
        set(main_source "${SB_TEST_MAIN_SOURCE}")
    endif()
    add_executable(${target} "${main_source}" ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE GTest::gtest)
    sushiblas_configure_test_target(${target})

    set(timeout ${SB_TEST_TIMEOUT_${ARG_SIZE}})
    set(gating_filter "*")
    if(ARG_QUARANTINED)
        list(JOIN ARG_QUARANTINED ":" quarantined_filter)
        set(gating_filter "*-${quarantined_filter}")
        gtest_discover_tests(${target}
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}"
            TEST_FILTER "${quarantined_filter}"
            PROPERTIES LABELS "${ARG_SIZE};quarantined" TIMEOUT ${timeout})
    endif()
    gtest_discover_tests(${target}
        DISCOVERY_MODE PRE_TEST
        WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}"
        TEST_FILTER "${gating_filter}"
        PROPERTIES LABELS "${ARG_SIZE}" TIMEOUT ${timeout})
endfunction()

# Builds the test program and the benchmark programs of one module.
#
#   sushiblas_add_test_program(MODULE <module>
#       [SIZE <small|medium|large>]
#       [QUARANTINED <Suite.Case>...]
#       [TESTS <file>...]
#       [BENCHMARKS <file>...])
#
# Each file is given relative to tests/<tier>/<module>/, the module's mirror. TESTS become
# the program `sushiblas_<module>_tests` when SB_BUILD_TESTS is on, sized by SIZE, which TESTS
# requires; each of BENCHMARKS becomes
# a program named after its file when SB_BUILD_BENCHMARKS is on.
function(sushiblas_add_test_program)
    cmake_parse_arguments(ARG "" "MODULE;SIZE" "QUARANTINED;TESTS;BENCHMARKS" ${ARGN})
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
        sushiblas_add_sized_test_program(${target} SIZE "${ARG_SIZE}"
            QUARANTINED ${ARG_QUARANTINED}
            SOURCES ${test_sources})
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
#       SIZE <small|medium|large>
#       [QUARANTINED <Suite.Case>...]
#       SOURCES <application source>...
#       TESTS <file>...)
#
# DIRECTORY is relative to applications/; SOURCES are relative to it and TESTS to its mirror
# tests/applications/<DIRECTORY>/. The program is `sushiblas_<name>_tests`, and it sees the
# application's own headers as the application does.
function(sushiblas_add_application_test_program)
    cmake_parse_arguments(ARG "" "NAME;DIRECTORY;SIZE" "QUARANTINED;SOURCES;TESTS" ${ARGN})
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
    sushiblas_add_sized_test_program(${target} SIZE "${ARG_SIZE}"
        QUARANTINED ${ARG_QUARANTINED}
        SOURCES ${application_sources} ${test_sources})
    target_include_directories(${target} PRIVATE "${application}")
endfunction()
