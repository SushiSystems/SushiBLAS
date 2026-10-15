# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

add_executable(quickstart_demo quickstart_demo.cpp)
target_link_libraries(quickstart_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(quickstart_demo)

add_executable(io_demo io_demo.cpp)
target_link_libraries(io_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(io_demo)

add_executable(adapter_demo
    adapter/adapter_demo.cpp
    adapter/sushiblas_adapter.cpp
)
target_link_libraries(adapter_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(adapter_demo)
sushiblas_add_application_test_program(
    NAME adapter
    SIZE medium
    DIRECTORY examples/adapter
    SOURCES sushiblas_adapter.cpp
    TESTS
        test_host_program.cpp
        test_status_mapping.cpp)

add_executable(errors_demo errors_demo.cpp)
target_link_libraries(errors_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(errors_demo)

add_executable(run_failure_demo run_failure_demo.cpp)
target_link_libraries(run_failure_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(run_failure_demo)

add_executable(recording_demo recording_demo.cpp)
target_link_libraries(recording_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(recording_demo)

add_executable(operation_families_demo operation_families_demo.cpp)
target_link_libraries(operation_families_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(operation_families_demo)

add_executable(graph_primitives_demo graph_primitives_demo.cpp)
target_link_libraries(graph_primitives_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(graph_primitives_demo)

# Ensure RPATH is set
set_target_properties(
    quickstart_demo
    io_demo
    adapter_demo
    errors_demo
    run_failure_demo
    recording_demo
    operation_families_demo
    graph_primitives_demo
    PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src"
)

# --- Post-Build: Copy the runtime DLL to the output directory ---
# hwloc is a private implementation detail of SushiRuntime's own build (linked
# PRIVATE via PkgConfig::HWLOC, BUILD_INTERFACE-only) and never reaches
# consumers, so only the runtime's own DLL needs copying here — matching
# sushiruntime's own examples/tests precedent exactly.
if(WIN32)
    add_custom_command(TARGET io_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:io_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET quickstart_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:quickstart_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET adapter_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:adapter_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET errors_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:errors_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET run_failure_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:run_failure_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET recording_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:recording_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET operation_families_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:operation_families_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET graph_primitives_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:graph_primitives_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )
endif()
