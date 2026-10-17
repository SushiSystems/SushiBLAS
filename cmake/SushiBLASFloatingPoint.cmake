# SushiBLASFloatingPoint.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASFloatingPoint.cmake — the floating-point model as a target.
#
# Single responsibility: define an INTERFACE target `sushiblas_floating_point`
# that compiles the tree with the precise floating-point model. icpx and icx
# default to the fast model, which lets the compiler and the OpenCL CPU JIT
# reorder and contract a kernel's sums, so two kernels that write one sum in
# one order can still round it differently. Under the precise model a sum
# rounds as the source writes it; a kernel that wants a fused multiply-add
# asks for one through Kernels::multiply_add. BuildPolicy.cmake applies it to
# this tree only; a consumer compiling the kernel headers chooses its own model.

add_library(sushiblas_floating_point INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(sushiblas_floating_point INTERFACE /fp:precise)
    else()
        target_compile_options(sushiblas_floating_point INTERFACE -ffp-model=precise)
    endif()
endif()
