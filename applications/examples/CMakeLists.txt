# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

add_executable(sushiblas_example example.cpp)
target_link_libraries(sushiblas_example PRIVATE sushiblas)
sushiblas_apply_build_policy(sushiblas_example)

add_executable(io_example io_example.cpp)
target_link_libraries(io_example PRIVATE sushiblas)
sushiblas_apply_build_policy(io_example)

add_executable(adapter_demo
    adapter/adapter_demo.cpp
    adapter/sushiblas_adapter.cpp
)
target_link_libraries(adapter_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(adapter_demo)

add_executable(errors_demo errors_demo.cpp)
target_link_libraries(errors_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(errors_demo)

add_executable(recording_demo recording_demo.cpp)
target_link_libraries(recording_demo PRIVATE sushiblas)
sushiblas_apply_build_policy(recording_demo)

# Ensure RPATH is set
set_target_properties(
    sushiblas_example
    io_example
    adapter_demo
    errors_demo
    recording_demo
    PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src"
)

# --- Post-Build: Copy the runtime DLL to the output directory ---
# hwloc is a private implementation detail of SushiRuntime's own build (linked
# PRIVATE via PkgConfig::HWLOC, BUILD_INTERFACE-only) and never reaches
# consumers, so only the runtime's own DLL needs copying here — matching
# sushiruntime's own examples/tests precedent exactly.
if(WIN32)
    add_custom_command(TARGET io_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:io_example>
        COMMENT "Copying runtime DLL to example directory..."
    )

    add_custom_command(TARGET sushiblas_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiblas_example>
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

    add_custom_command(TARGET recording_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:recording_demo>
        COMMENT "Copying runtime DLL to example directory..."
    )
endif()
