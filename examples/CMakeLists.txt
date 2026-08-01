add_executable(sushiblas_example example.cpp)
target_link_libraries(sushiblas_example PRIVATE sushiblas)

add_executable(io_example io_example.cpp)
target_link_libraries(io_example PRIVATE sushiblas)

# Ensure RPATH is set
set_target_properties(sushiblas_example io_example PROPERTIES
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
endif()
