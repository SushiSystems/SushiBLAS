cmake_minimum_required(VERSION 3.20)

# --- Define Executable (main app) ---
add_executable(sushiblas_app main.cpp)

# --- Include Directories ---
target_include_directories(sushiblas_app PUBLIC 
    "${CMAKE_SOURCE_DIR}/include"
)

# --- Link Directories ---
target_link_directories(sushiblas_app PRIVATE "${CMAKE_SOURCE_DIR}/lib")

# --- Link Libraries ---
# Link against the pre-compiled SushiRuntime.lib and SYCL for the headers/compiler support
target_link_libraries(sushiblas_app PRIVATE
    sushiruntime
    IntelSYCL::SYCL_CXX
)

# --- Properties ---
set_target_properties(sushiblas_app PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# --- Post-Build: Copy DLLs to output directory ---
add_custom_command(TARGET sushiblas_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/lib/sushiruntime.dll"
        "${CMAKE_SOURCE_DIR}/lib/hwloc-15.dll"
        $<TARGET_FILE_DIR:sushiblas_app>
    COMMENT "Copying runtime DLLs to application directory..."
)
