cmake_minimum_required(VERSION 3.20)

# --- vcpkg Integration ---
# Check if VCPKG_ROOT is defined in env if not passed via CMake
if(NOT DEFINED VCPKG_ROOT AND DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_ROOT "$ENV{VCPKG_ROOT}" CACHE PATH "vcpkg root directory")
endif()

# Automatically set toolchain file if VCPKG_ROOT is available
if(DEFINED VCPKG_ROOT AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(EXISTS "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file")
    endif()
endif()

if(DEFINED VCPKG_ROOT)
     message(STATUS ">>> Using vcpkg from: ${VCPKG_ROOT}")
endif()

# --- Intel oneAPI Setup ---
if(NOT DEFINED CMAKE_CXX_COMPILER)
    # Try to locate Intel oneAPI from environment
    if(NOT DEFINED ONEAPI_ROOT AND DEFINED ENV{ONEAPI_ROOT})
        set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}" CACHE PATH "Intel oneAPI root directory")
    endif()

    if(DEFINED ONEAPI_ROOT)
        set(INTEL_ONEAPI_DIR "${ONEAPI_ROOT}/compiler/latest")
        if(WIN32 AND EXISTS "${INTEL_ONEAPI_DIR}/bin/icx-cl.exe")
            set(CMAKE_CXX_COMPILER "${INTEL_ONEAPI_DIR}/bin/icx-cl.exe" CACHE FILEPATH "CXX compiler")
            set(CMAKE_C_COMPILER "${INTEL_ONEAPI_DIR}/bin/icx-cl.exe" CACHE FILEPATH "C compiler")
        elseif(UNIX AND EXISTS "${INTEL_ONEAPI_DIR}/linux/bin/icpx")
            set(CMAKE_CXX_COMPILER "${INTEL_ONEAPI_DIR}/linux/bin/icpx" CACHE FILEPATH "CXX compiler")
            set(CMAKE_C_COMPILER "${INTEL_ONEAPI_DIR}/linux/bin/icx" CACHE FILEPATH "C compiler")
        endif()

        if(DEFINED CMAKE_CXX_COMPILER)
            message(STATUS ">>> Using Intel Compiler: ${CMAKE_CXX_COMPILER}")
            
            # Intel runtime libs (needed for compiler checks)
            if(WIN32)
                set(INTEL_LIB_PATH "${INTEL_ONEAPI_DIR}/lib")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LIBPATH:\"${INTEL_LIB_PATH}\"")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
                list(APPEND CMAKE_PREFIX_PATH "${INTEL_ONEAPI_DIR}/lib/cmake/IntelSYCL")
            else()
                # On Linux, flags are naturally handled by icpx/setvars.sh
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
            endif()
        endif()
    endif()
endif()

project(SushiBLAS VERSION 0.1.0 LANGUAGES CXX C)

# --- Policies ---
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# --- Configuration ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Compiler Flags ---
if(MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Windows / Intel icx-cl
    add_compile_options(/W4 /permissive-)
    
    # Release / Maximum Performance Optimizations
    add_compile_options($<$<CONFIG:Release>:/O2>)         # Maximum speed optimization
    add_compile_options($<$<CONFIG:Release>:/fp:fast>)    # Fast floating-point math
    add_compile_options($<$<CONFIG:Release>:/arch:AVX2>)  # Enable AVX2 instructions (Works on AMD)
    
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    # Linux / GCC / Clang
    add_compile_options(-O3 -march=native -Wall -Wextra -Werror -pedantic)
endif()

# --- Dependencies ---

# 1. SYCL (Intel DPC++ or DataC++)
# Intel oneAPI DPC++ compiler is required for SYCL support.
# IntelSYCL is the modern replacement for IntelDPCPP.
find_package(IntelSYCL REQUIRED)
find_package(OpenCL REQUIRED)

# --- Subdirectories ---
add_subdirectory(src)
add_subdirectory(examples)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
