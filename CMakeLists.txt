cmake_minimum_required(VERSION 3.20)

# --- Force Authoritative vcpkg Path (Fixes VS Hijacking) ---
if(WIN32)
    set(VCPKG_ROOT "C:/vcpkg" CACHE PATH "" FORCE)
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "" FORCE)
    message(STATUS ">>> Forced VCPKG_ROOT to: ${VCPKG_ROOT}")
endif()

# --- vcpkg Classic Mode ---
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# --- Compiler Selection ---
if(WIN32 AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(INTEL_ONEAPI_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
    set(CMAKE_CXX_COMPILER "${INTEL_ONEAPI_DIR}/bin/icx-cl.exe" CACHE FILEPATH "CXX compiler")
    # Intel runtime libs (fixes LNK1104 during compiler check)
    set(INTEL_LIB_PATH "${INTEL_ONEAPI_DIR}/lib")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LIBPATH:\"${INTEL_LIB_PATH}\"")
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    list(APPEND CMAKE_PREFIX_PATH "${INTEL_ONEAPI_DIR}/lib/cmake/IntelSYCL")
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
    add_compile_options($<$<CONFIG:Release>:/O2>)
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

# --- Testing ---
enable_testing()
add_subdirectory(tests)
# add_subdirectory(examples)