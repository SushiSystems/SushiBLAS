# SushiBLASTests.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# SushiBLASTests.cmake — the one shape of a module's tests and of a tier's test program.
#
# Single responsibility: build the test objects and the benchmark programs of one module from
# its mirror under tests/<tier>/<module>/, link the test objects of each tier into one program,
# and register its tests, in shards, under their size. Every module goes through the same function, so every
# program is linked, labelled and placed the same way. A program repeats the device link of the
# library it links, so one program per tier links that device code once instead of once per
# module.
#
# A test's size is the prefix of its GoogleTest suite name: Small_, Medium_ or Large_. Each size
# is registered by its own filter, in shards of N cases, with its label and its timeout per case. A suite with none of the three
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
    # tests/ is an include root, so a test reaches the shared helpers as "support/<name>.hpp".
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/tests")
    sushiblas_apply_build_policy(${target})
    get_target_property(type ${target} TYPE)
    if(NOT type STREQUAL "EXECUTABLE")
        return()
    endif()
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${SB_TEST_OUTPUT_DIR}")
    # hwloc is private to SushiRuntime's own build and never reaches a consumer. gtest.dll is
    # copied from the configuration the program links: a Debug tree links GTest's Release build,
    # and vcpkg's own copy, which runs before this one, puts the Debug DLL there.
    if(WIN32)
        set(gtest_dll "")
        if(TARGET GTest::gtest)
            set(gtest_dll "$<TARGET_FILE:GTest::gtest>")
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::shared_library>"
                "$<TARGET_FILE:SushiLog::SushiLog>"
                ${gtest_dll}
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying sushiruntime.dll, sushilog.dll and gtest.dll to the test directory")
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

# The cases one CTest test runs, per size. A process pays its start-up (the runtime, the device
# discovery, the logger) once, so a case run alone is mostly start-up. A shard runs its share of a
# program's cases in one process, and the share is sized to the cost of a case of its size. A
# small case is a few milliseconds of work, so 300 of them are a few seconds and the run still
# spreads over every core; a medium case reaches a device or a thread and takes up to 5 s, so 25
# of them keep a shard well under a minute; a large case takes up to 120 s and runs alone.
set(SB_SHARD_CASES_SMALL 300)
set(SB_SHARD_CASES_MEDIUM 25)
set(SB_SHARD_CASES_LARGE 1)

# Where each shard writes its per-case GoogleTest XML; the time reader reads the folder, so the
# per-case limits of the testing skill still apply to a case run inside a shard.
set(SB_GTEST_RESULTS_DIR "${PROJECT_BINARY_DIR}/test_results/gtest")

# Sets <out_var> to the cases per shard of the size with the given prefix.
function(sushiblas_cases_per_shard out_var prefix)
    string(TOUPPER "${prefix}" upper)
    string(REGEX REPLACE "_$" "" upper "${upper}")
    set(${out_var} ${SB_SHARD_CASES_${upper}} PARENT_SCOPE)
endfunction()

# Counts the cases of the given sources per size, and keeps the counts and the suites under
# <key>. A TEST_P or TYPED_TEST counts once however many instances it gets, so a count is a floor.
#
#   sushiblas_record_cases(<key> <source>...)
function(sushiblas_record_cases key)
    foreach(size IN LISTS SB_TEST_SIZES)
        string(REPLACE "|" ";" parts "${size}")
        list(GET parts 0 prefix)
        set(count_${prefix} 0)
    endforeach()
    set(suites "")
    foreach(source IN LISTS ARGN)
        file(READ "${source}" text)
        string(REGEX MATCHALL "(TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)[ \t]*[(][ \t\r\n]*[A-Za-z0-9_]+"
            declarations "${text}")
        foreach(declaration IN LISTS declarations)
            string(REGEX REPLACE "^[^(]*[(][ \t\r\n]*" "" suite "${declaration}")
            list(APPEND suites "${suite}")
            if(suite MATCHES "^(Small_|Medium_|Large_)")
                math(EXPR count_${CMAKE_MATCH_1} "${count_${CMAKE_MATCH_1}} + 1")
            endif()
        endforeach()
    endforeach()
    foreach(size IN LISTS SB_TEST_SIZES)
        string(REPLACE "|" ";" parts "${size}")
        list(GET parts 0 prefix)
        set_property(GLOBAL PROPERTY SB_CASES_${key}_${prefix} ${count_${prefix}})
    endforeach()
    set_property(GLOBAL PROPERTY SB_SUITES_${key} ${suites})
endfunction()

# Gives <key> the cases and suites recorded under each of the object keys that follow.
#
#   sushiblas_sum_cases(<key> <object key>...)
function(sushiblas_sum_cases key)
    set(suites "")
    foreach(size IN LISTS SB_TEST_SIZES)
        string(REPLACE "|" ";" parts "${size}")
        list(GET parts 0 prefix)
        set(total 0)
        foreach(object IN LISTS ARGN)
            get_property(count GLOBAL PROPERTY SB_CASES_${object}_${prefix})
            math(EXPR total "${total} + ${count}")
        endforeach()
        set_property(GLOBAL PROPERTY SB_CASES_${key}_${prefix} ${total})
    endforeach()
    foreach(object IN LISTS ARGN)
        get_property(object_suites GLOBAL PROPERTY SB_SUITES_${object})
        list(APPEND suites ${object_suites})
    endforeach()
    set_property(GLOBAL PROPERTY SB_SUITES_${key} ${suites})
endfunction()

# Registers one CTest test that runs <target> with <filter>, as shard <index> of <shards>.
# GoogleTest splits the filtered cases round-robin by GTEST_TOTAL_SHARDS and GTEST_SHARD_INDEX.
# The test writes its per-case XML under SB_GTEST_RESULTS_DIR. The shuffle and its seed reach it
# through GTEST_SHUFFLE and GTEST_RANDOM_SEED of the caller's environment, and a death test starts
# its child outside the sharding, so both keep working inside a shard.
function(sushiblas_add_shard_test target name filter labels timeout index shards)
    add_test(NAME ${name}
        COMMAND ${target} --gtest_filter=${filter}
            --gtest_output=xml:${SB_GTEST_RESULTS_DIR}/${name}.xml
        WORKING_DIRECTORY "${SB_TEST_OUTPUT_DIR}")
    set_tests_properties(${name} PROPERTIES
        LABELS "${labels}"
        TIMEOUT ${timeout}
        ENVIRONMENT "GTEST_TOTAL_SHARDS=${shards};GTEST_SHARD_INDEX=${index}")
endfunction()

# Registers the cases of <target> of one size in shards, with the size's label and a timeout of
# the case timeout times the cases of a shard. The quarantined cases of the size run apart in one
# test that also carries `quarantined`, and a GoogleTest filter after '-' leaves them out of the
# shards. <size> is one SB_TEST_SIZES entry; the names after it are the program's quarantined
# `Suite.Case` names.
function(sushiblas_register_size_shards target size)
    string(REPLACE "|" ";" parts "${size}")
    list(GET parts 0 prefix)
    list(GET parts 1 label)
    list(GET parts 2 case_timeout)
    sushiblas_cases_per_shard(per_shard ${prefix})
    get_property(cases GLOBAL PROPERTY SB_CASES_${target}_${prefix})

    # An instantiated suite is named <prefix>/<suite>, so its pattern carries the slash.
    set(quarantined "")
    foreach(name IN LISTS ARGN)
        if(name MATCHES "^${prefix}")
            list(APPEND quarantined "${name}" "*/${name}")
        endif()
    endforeach()
    set(filter "${prefix}*:*/${prefix}*")
    if(quarantined)
        list(LENGTH quarantined quarantined_entries)
        math(EXPR quarantined_cases "${quarantined_entries} / 2")
        list(JOIN quarantined ":" quarantined_filter)
        string(APPEND filter "-${quarantined_filter}")
        math(EXPR quarantined_timeout "${case_timeout} * ${quarantined_cases}")
        sushiblas_add_shard_test(${target} ${target}.${label}_quarantined "${quarantined_filter}"
            "${label};quarantined" ${quarantined_timeout} 0 1)
        math(EXPR cases "${cases} - ${quarantined_cases}")
    endif()
    if(cases LESS 1)
        return()
    endif()

    math(EXPR shards "(${cases} + ${per_shard} - 1) / ${per_shard}")
    math(EXPR shard_cases "(${cases} + ${shards} - 1) / ${shards}")
    math(EXPR timeout "${case_timeout} * ${shard_cases}")
    math(EXPR last "${shards} - 1")
    foreach(index RANGE ${last})
        sushiblas_add_shard_test(${target} ${target}.${label}_shard_${index}of${shards}
            "${filter}" "${label}" ${timeout} ${index} ${shards})
    endforeach()
endfunction()

# Registers the tests of a GoogleTest program in shards, one block per size. The cases of the
# program were recorded under its target name. Each name in QUARANTINED, a GoogleTest
# `Suite.Case` name, runs apart with the label `quarantined` too.
#
#   sushiblas_register_sized_tests(<target> [QUARANTINED <Suite.Case>...])
function(sushiblas_register_sized_tests target)
    cmake_parse_arguments(ARG "" "" "QUARANTINED" ${ARGN})
    foreach(size IN LISTS SB_TEST_SIZES)
        sushiblas_register_size_shards(${target} "${size}" ${ARG_QUARANTINED})
    endforeach()
endfunction()

# Builds one GoogleTest program from the sources given and registers its tests in shards by size. The
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
    sushiblas_record_cases(${target} ${ARG_TEST_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
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
    sushiblas_record_cases(${target} ${ARG_TEST_SOURCES} ${ARG_DEVICE_TEST_SOURCES})
    set_property(GLOBAL APPEND PROPERTY SB_TEST_TIERS ${tier})
    set_property(GLOBAL APPEND PROPERTY SB_TEST_OBJECTS_${tier} ${target})
    set_property(GLOBAL APPEND PROPERTY SB_TEST_QUARANTINED_${tier} ${ARG_QUARANTINED})
endfunction()

# Builds sushiblas_<tier>_tests from the test objects sushiblas_add_test_objects() recorded for
# each tier, with the shared main(), and registers its tests in shards by size. Called once, after every
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
        sushiblas_sum_cases(${program} ${objects})
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
#       [BENCHMARKS <file>...] [DEVICE_BENCHMARKS <file>...]
#       [COMPILE_CHECKS <file>...])
#
# Each file is given relative to tests/<tier>/<module>/, the module's mirror. TESTS and
# DEVICE_TESTS become the object library `sushiblas_<module>_test_objects`, linked into the
# program `sushiblas_<tier>_tests`, when SB_BUILD_TESTS is on; each of
# BENCHMARKS and DEVICE_BENCHMARKS becomes a program named after its file when
# SB_BUILD_BENCHMARKS is on. The DEVICE_ lists name the files that instantiate a kernel.
# COMPILE_CHECKS become `sushiblas_<module>_compile_checks`, compiled and never run.
function(sushiblas_add_test_program)
    cmake_parse_arguments(ARG "" "MODULE"
        "QUARANTINED;TESTS;DEVICE_TESTS;BENCHMARKS;DEVICE_BENCHMARKS;COMPILE_CHECKS" ${ARGN})
    if("${ARG_MODULE}" STREQUAL ""
       OR (NOT ARG_TESTS AND NOT ARG_DEVICE_TESTS AND NOT ARG_BENCHMARKS
           AND NOT ARG_DEVICE_BENCHMARKS AND NOT ARG_COMPILE_CHECKS))
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

    # A compile check holds static_asserts only: the build compiles it and no test runner lists it.
    if(SB_BUILD_TESTS AND ARG_COMPILE_CHECKS)
        set(target "sushiblas_${ARG_MODULE}_compile_checks")
        sushiblas_paths_under(compile_sources "${mirror}" ${ARG_COMPILE_CHECKS})
        add_library(${target} OBJECT ${compile_sources})
        sushiblas_configure_test_target(${target})
        sushiblas_apply_sycl(${target})
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
