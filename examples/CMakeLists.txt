add_executable(sushiblas_example example.cpp)
target_link_libraries(sushiblas_example PRIVATE sushiblas IntelSYCL::SYCL_CXX)

add_executable(io_example io_example.cpp)
target_link_libraries(io_example PRIVATE sushiblas IntelSYCL::SYCL_CXX)

# Ensure RPATH is set
set_target_properties(sushiblas_example io_example PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src"
)

# --- Post-Build: Copy DLLs to output directory ---
if(WIN32)
    add_custom_command(TARGET io_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/lib/sushiruntime.dll"
            "${CMAKE_SOURCE_DIR}/lib/hwloc-15.dll"
            $<TARGET_FILE_DIR:io_example>
        COMMENT "Copying runtime DLLs to example directory..."
    )
    
    add_custom_command(TARGET sushiblas_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/lib/sushiruntime.dll"
            "${CMAKE_SOURCE_DIR}/lib/hwloc-15.dll"
            $<TARGET_FILE_DIR:sushiblas_example>
        COMMENT "Copying runtime DLLs to example directory..."
    )
endif()
