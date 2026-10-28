# SushiBLASTests.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASTests.cmake — the one shape of a module's tests and of a tier's test program.
#
# Single responsibility: build the test objects and the benchmark programs of one module from
# its mirror under tests/<tier>/<module>/, link the test objects of each tier into one program,
# and register its tests under their size. Every module goes through the same function, so every
# program is linked, labelled and placed the same way. A program repeats the device link of the
# library it links, so one program per tier links that device code once instead of once per
# module.
#
# A test's size is the prefix of its GoogleTest suite name: Small_, Medium_ or Large_. Each size
# is registered by its own filter, with its label and its timeout. A suite with none of the three
# prefixes stops the configure. A program whose suites are all Small_ runs without the shared
# runtime environment, so it starts no worker thread and opens no device; a tier program's main()
# leaves the environment out when its filter selects only Small_ suites. A test listed as
# quarantined keeps its size and also carries the label `quarantined`, which takes it out of
# every gating stage. A benchmark program carries the target label `benchmark` and is never
# registered with CTest: it measures, it does not pass or fail.

# Every program lands in one folder, so the runtime DLL is copied to one place and
# `sb run <program>` finds each of them there.
set(SB_TEST_OUTPUT_DIR "${PROJECT_BINARY_DIR}/tests")
# The main() of a program with a Medium_ or Large_ suite, which owns the shared runtime
# environment, and of a program whose suites are all Small_, which owns none. Both belong to the test_runner module and are compiled into
# each program, never into the library.
set(SB_TEST_MAIN_SOURCE "${PROJECT_SOURCE_DIR}/modules/foundation/test_runner/source/test_main.cpp")
set(SB_SMALL_TEST_MAIN_SOURCE
    "${PROJECT_SOURCE_DIR}/modules/foundation/test_runner/source/small_test_main.cpp")
# Each size as prefix|label|timeout in seconds.
set(SB_TEST_SIZES "Small_|small|60" "Medium_|medium|300" "Large_|large|900")

# Links a program to the library and the INTERNAL modules' include roots, applies the build
# policy, places it in SB_TEST_OUTPUT_DIR and copies the runtime DLL beside it on Windows.
# A test may include a kernel a consumer cannot, which is how a kernel variant the device
# profile would not pick on this machine still gets run.
# An object library gets the links and the policy only: it has no file to place.
function(sushiblas_configure_test_target target)
    target_link_libraries(${target} PRIVATE
        sushiblas
        sushiblas_internal_headers)
    sushiblas_apply_build_policy(${target})
    get_target_property(type ${target} TYPE)
    if(NOT type STREQUAL "EXECUTABLE")
        return()
    endif()
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${SB_TEST_OUTPUT_DIR}")
    # hwloc is private to SushiRuntime's own build and never reaches a consumer.
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::shared_library>"
                "$<TARGET_FILE:SushiLog::SushiLog>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying sushiruntime.dll and sushilog.dll to the test directory")
        if(SB_ENABLE_PROFILER)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:SushiProfiler::SushiProfiler>"
                    $<TARGET_FILE_DIR:${target}>
                COMMENT "Copying sushiprofiler.dll to the test directory")
        endif()
    endif()
endfunction()

# Sets <out_var> to the GoogleTest suite names the TEST, TEST_F, TEST_P, TYPED_TEST and
# TYPED_TEST_P macros of the given sources declare.
function(sushiblas_test_suites_of out_var)
    set(suites "")
    foreach(source IN LISTS ARGN)
        file(READ "${source}" text)
        string(REGEX MATCHALL "TEST(_F|_P)?[ \t]*[(][ \t\r\n]*[A-Za-z0-9_]+" uses "${text}")
        foreach(use IN LISTS uses)
            string(REGEX REPLACE "^[^(]*[(][ \t\r\n]*" "" suite "${use}")
            list(APPEND suites "${suite}")
        endforeach()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    endforeach()
    list(REMOVE_DUPLICATES suites)
    set(${out_var} "${suites}" PARENT_SCOPE)
endfunction()

# Sets <out_var> to TRUE when every suite the given sources declare is Small_, and stops the
# configure at a suite with no size. <target> names the program in the message.
function(sushiblas_check_test_sizes out_var target)
    sushiblas_test_suites_of(suites ${ARGN})
    set(all_small TRUE)
    foreach(suite IN LISTS suites)
        if(NOT suite MATCHES "^(Small|Medium|Large)_")
            message(FATAL_ERROR
                "${target}: the suite ${suite} has no size; name it Small_, Medium_ or Large_.")
        endif()
        if(NOT suite MATCHES "^Small_")
            set(all_small FALSE)
        endif()
    endforeach()
    set(${out_var} ${all_small} PARENT_SCOPE)
endfunction()

# Registers the tests of a GoogleTest program by size. Each name in QUARANTINED, a GoogleTest
# `Suite.Case` name, is registered with the label `quarantined` too.
#
#   sushiblas_register_sized_tests(<target> [QUARANTINED <Suite.Case>...])
function(sushiblas_register_sized_tests target)
    cmake_parse_arguments(ARG "" "" "QUARANTINED" ${ARGN})
    foreach(size IN LISTS SB_TEST_SIZES)
        string(REPLACE "|" ";" parts "${size}")
        list(GET parts 0 prefix)
        list(GET parts 1 label)
        list(GET parts 2 timeout)

        # A GoogleTest filter after '-' excludes, so the quarantined tests register apart.
        set(quarantined "")
        foreach(name IN LISTS ARG_QUARANTINED)
            if(name MATCHES "^${prefix}")
                list(APPEND quarantined "${name}")
            endif()
        endforeach()
        list(JOIN quarantined ":" quarantined_filter)

        set(filter "${prefix}*")
        if(quarantined_filter)
            string(APPEND filter "-${quarantined_filter}")
            gtest_discover_tests(${target}
                DISCOVERY_MODE PRE_TEST
                WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}"
                TEST_FILTER "${quarantined_filter}"
                PROPERTIES LABELS "${label}\;quarantined" TIMEOUT ${timeout})
        endif()
        gtest_discover_tests(${target}
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}"
            TEST_FILTER "${filter}"
            PROPERTIES LABELS "${label}" TIMEOUT ${timeout})
    endforeach()
endfunction()

# Builds one GoogleTest program from the sources given and registers its tests by size. The
# program gets the main() without the runtime when every suite is Small_.
#
#   sushiblas_add_sized_test_program(<target> [QUARANTINED <Suite.Case>...]
#       [SOURCES <file>...] [DEVICE_SOURCES <file>...]
#       [TEST_SOURCES <file>...] [DEVICE_TEST_SOURCES <file>...])
#
# TEST_SOURCES and DEVICE_TEST_SOURCES are scanned for suites; SOURCES and DEVICE_SOURCES are
# compiled with them and not scanned. The DEVICE_ lists name the units that instantiate a kernel.
function(sushiblas_add_sized_test_program target)
    cmake_parse_arguments(ARG "" ""
        "QUARANTINED;SOURCES;DEVICE_SOURCES;TEST_SOURCES;DEVICE_TEST_SOURCES" ${ARGN})
    sushiblas_check_test_sizes(all_small ${target} ${ARG_TEST_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
    if(all_small)
        set(main_source "${SB_SMALL_TEST_MAIN_SOURCE}")
    else()
        set(main_source "${SB_TEST_MAIN_SOURCE}")
    endif()
    add_executable(${target} "${main_source}" ${ARG_SOURCES} ${ARG_DEVICE_SOURCES}
        ${ARG_TEST_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
    target_link_libraries(${target} PRIVATE GTest::gtest)
    sushiblas_configure_test_target(${target})
    sushiblas_apply_sycl(${target} DEVICE_SOURCES ${ARG_DEVICE_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
    sushiblas_register_sized_tests(${target} QUARANTINED ${ARG_QUARANTINED})
endfunction()

# Builds the test objects of one module, an object library, and records them, with the module's
# quarantined names, for the program of its tier.
#
#   sushiblas_add_test_objects(<target> <tier> [QUARANTINED <Suite.Case>...]
#       [TEST_SOURCES <file>...] [DEVICE_TEST_SOURCES <file>...])
function(sushiblas_add_test_objects target tier)
    cmake_parse_arguments(ARG "" "" "QUARANTINED;TEST_SOURCES;DEVICE_TEST_SOURCES" ${ARGN})
    sushiblas_check_test_sizes(all_small ${target} ${ARG_TEST_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
    add_library(${target} OBJECT ${ARG_TEST_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
    target_link_libraries(${target} PRIVATE GTest::gtest)
    sushiblas_configure_test_target(${target})
    sushiblas_apply_sycl(${target} DEVICE_SOURCES ${ARG_DEVICE_TEST_SOURCES})
    set_property(GLOBAL APPEND PROPERTY SB_TEST_TIERS ${tier})
    set_property(GLOBAL APPEND PROPERTY SB_TEST_OBJECTS_${tier} ${target})
    set_property(GLOBAL APPEND PROPERTY SB_TEST_QUARANTINED_${tier} ${ARG_QUARANTINED})
endfunction()

# Builds sushiblas_<tier>_tests from the test objects sushiblas_add_test_objects() recorded for
# each tier, with the shared main(), and registers its tests by size. Called once, after every
# module is read.
function(sushiblas_add_tier_test_programs)
    get_property(tiers GLOBAL PROPERTY SB_TEST_TIERS)
    list(REMOVE_DUPLICATES tiers)
    foreach(tier IN LISTS tiers)
        get_property(objects GLOBAL PROPERTY SB_TEST_OBJECTS_${tier})
        get_property(quarantined GLOBAL PROPERTY SB_TEST_QUARANTINED_${tier})
        set(program sushiblas_${tier}_tests)
        add_executable(${program} "${SB_TEST_MAIN_SOURCE}")
        target_link_libraries(${program} PRIVATE ${objects} GTest::gtest)
        sushiblas_configure_test_target(${program})
        sushiblas_apply_sycl(${program})
        sushiblas_register_sized_tests(${program} QUARANTINED ${quarantined})
    endforeach()
endfunction()

# Sets <out_var> to each file given, prefixed with <folder>/, or to an empty list.
function(sushiblas_paths_under out_var folder)
    set(paths "")
    foreach(file IN LISTS ARGN)
        list(APPEND paths "${folder}/${file}")
    endforeach()
    set(${out_var} "${paths}" PARENT_SCOPE)
endfunction()

# Builds the test program and the benchmark programs of one module.
#
#   sushiblas_add_test_program(MODULE <module>
#       [QUARANTINED <Suite.Case>...]
#       [TESTS <file>...] [DEVICE_TESTS <file>...]
#       [BENCHMARKS <file>...] [DEVICE_BENCHMARKS <file>...])
#
# Each file is given relative to tests/<tier>/<module>/, the module's mirror. TESTS and
# DEVICE_TESTS become the object library `sushiblas_<module>_test_objects`, linked into the
# program `sushiblas_<tier>_tests`, when SB_BUILD_TESTS is on; each of
# BENCHMARKS and DEVICE_BENCHMARKS becomes a program named after its file when
# SB_BUILD_BENCHMARKS is on. The DEVICE_ lists name the files that instantiate a kernel.
function(sushiblas_add_test_program)
    cmake_parse_arguments(ARG "" "MODULE"
        "QUARANTINED;TESTS;DEVICE_TESTS;BENCHMARKS;DEVICE_BENCHMARKS" ${ARGN})
    if("${ARG_MODULE}" STREQUAL ""
       OR (NOT ARG_TESTS AND NOT ARG_DEVICE_TESTS AND NOT ARG_BENCHMARKS
           AND NOT ARG_DEVICE_BENCHMARKS))
        message(FATAL_ERROR
            "sushiblas_add_test_program(): MODULE and a test or benchmark list are required.")
    endif()
    sushiblas_module_layer("${ARG_MODULE}" layer)
    if("${layer}" STREQUAL "")
        message(FATAL_ERROR "sushiblas_add_test_program(): ${ARG_MODULE} has no tier in SUSHIBLAS_MODULE_LAYERS.")
    endif()
    set(mirror "${PROJECT_SOURCE_DIR}/tests/${layer}/${ARG_MODULE}")

    if(SB_BUILD_TESTS AND (ARG_TESTS OR ARG_DEVICE_TESTS))
        set(target "sushiblas_${ARG_MODULE}_test_objects")
        sushiblas_paths_under(test_sources "${mirror}" ${ARG_TESTS})
        sushiblas_paths_under(device_test_sources "${mirror}" ${ARG_DEVICE_TESTS})
        sushiblas_add_test_objects(${target} ${layer}
            QUARANTINED ${ARG_QUARANTINED}
            TEST_SOURCES ${test_sources}
            DEVICE_TEST_SOURCES ${device_test_sources})
    endif()

    if(SB_BUILD_BENCHMARKS)
        foreach(source IN LISTS ARG_BENCHMARKS ARG_DEVICE_BENCHMARKS)
            get_filename_component(benchmark "${source}" NAME_WE)
            add_executable(${benchmark} "${mirror}/${source}")
            sushiblas_configure_test_target(${benchmark})
            if(source IN_LIST ARG_DEVICE_BENCHMARKS)
                sushiblas_apply_sycl(${benchmark} DEVICE_SOURCES "${mirror}/${source}")
            else()
                sushiblas_apply_sycl(${benchmark})
            endif()
            set_property(TARGET ${benchmark} PROPERTY LABELS benchmark)
        endforeach()
    endif()
endfunction()

# Builds the test program of one application from its mirror under tests/applications/.
#
#   sushiblas_add_application_test_program(NAME <name> DIRECTORY <application folder>
#       [QUARANTINED <Suite.Case>...]
#       [SOURCES <application source>...] [DEVICE_SOURCES <application source>...]
#       [TESTS <file>...] [DEVICE_TESTS <file>...])
#
# DIRECTORY is relative to applications/; SOURCES and DEVICE_SOURCES are relative to it, and
# TESTS and DEVICE_TESTS to its mirror tests/applications/<DIRECTORY>/. The DEVICE_ lists name
# the files that instantiate a kernel. The program is `sushiblas_<name>_tests`, and it sees the
# application's own headers as the application does.
function(sushiblas_add_application_test_program)
    cmake_parse_arguments(ARG "" "NAME;DIRECTORY"
        "QUARANTINED;SOURCES;DEVICE_SOURCES;TESTS;DEVICE_TESTS" ${ARGN})
    if("${ARG_NAME}" STREQUAL "" OR "${ARG_DIRECTORY}" STREQUAL ""
       OR (NOT ARG_TESTS AND NOT ARG_DEVICE_TESTS))
        message(FATAL_ERROR
            "sushiblas_add_application_test_program(): NAME, DIRECTORY and a test list are "
            "required.")
    endif()
    if(NOT SB_BUILD_TESTS)
        return()
    endif()
    set(application "${PROJECT_SOURCE_DIR}/applications/${ARG_DIRECTORY}")
    set(mirror "${PROJECT_SOURCE_DIR}/tests/applications/${ARG_DIRECTORY}")
    sushiblas_paths_under(application_sources "${application}" ${ARG_SOURCES})
    sushiblas_paths_under(device_application_sources "${application}" ${ARG_DEVICE_SOURCES})
    sushiblas_paths_under(test_sources "${mirror}" ${ARG_TESTS})
    sushiblas_paths_under(device_test_sources "${mirror}" ${ARG_DEVICE_TESTS})
    set(target "sushiblas_${ARG_NAME}_tests")
    sushiblas_add_sized_test_program(${target}
        QUARANTINED ${ARG_QUARANTINED}
        SOURCES ${application_sources}
        DEVICE_SOURCES ${device_application_sources}
        TEST_SOURCES ${test_sources}
        DEVICE_TEST_SOURCES ${device_test_sources})
    target_include_directories(${target} PRIVATE "${application}")
endfunction()
