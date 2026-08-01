cmake_minimum_required(VERSION 3.20)

# SushiBLAS — a head; SushiRuntime is the plugged-in component (the battery).
# Mirrors SushiEngine's own root CMakeLists.txt ordering exactly (both are
# head modules over the same backbone): Vcpkg + Toolchain run BEFORE
# project() to shape the toolchain file and pick the default SYCL compiler
# ahead of project()'s probe; everything else (the runtime dependency, the
# SYCL target policy) runs AFTER. See cmake/ for each stage.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Pre-project: toolchain shaping -------------------------------------------
include(Vcpkg)      # (Windows) locate vcpkg + set the toolchain file
include(Toolchain)  # pick the default SYCL compiler (clang++) before the probe

project(SushiBLAS VERSION 0.1.0 LANGUAGES CXX C)

# --- Post-project: runtime dependency (D3 layered resolution, SB-0), SYCL policy
include(Runtime)     # add_subdirectory(sushiruntime) -> SushiRuntime::SushiRuntime,
                      # add_sycl_to_target(), SR_SYCL_TOOLCHAIN
include(SyclTarget)  # sushiblas_apply_sycl() helper

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

# --- Subdirectories ---
add_subdirectory(src)
add_subdirectory(examples)

# --- Testing ---
enable_testing()
add_subdirectory(tests)
