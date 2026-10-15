# SushiBLASSanitizers.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASSanitizers.cmake — the project's sanitizer policy as a target.
#
# Single responsibility: define an INTERFACE target `sushiblas_sanitizers` that carries the
# sanitizers SB_SANITIZE names. Empty, the default, adds nothing. `sb build --sanitize` sets it:
# "address,undefined" for the AddressSanitizer and UndefinedBehaviorSanitizer lane, "thread" for
# the ThreadSanitizer lane. Every target links the target unconditionally through
# BuildPolicy.cmake, so a lane changes no target's CMakeLists.txt.
#
# The sanitizers instrument host code only. `-Xarch_host` keeps them off the SYCL device
# compile, which has no sanitizer runtime for every device a build targets. An undefined
# behaviour report stops the program, so it fails the test instead of passing with a log line.

set(SB_SANITIZE "" CACHE STRING
    "Sanitizers for host code, as -fsanitize= takes them: empty, address,undefined or thread.")

add_library(sushiblas_sanitizers INTERFACE)

if(SB_SANITIZE)
    if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        message(FATAL_ERROR "SB_SANITIZE=${SB_SANITIZE} needs a GNU-style compiler driver.")
    endif()
    target_compile_options(sushiblas_sanitizers INTERFACE
        "SHELL:-Xarch_host -fsanitize=${SB_SANITIZE}"
        "SHELL:-Xarch_host -fno-omit-frame-pointer")
    target_link_options(sushiblas_sanitizers INTERFACE "-fsanitize=${SB_SANITIZE}")
    if(SB_SANITIZE MATCHES "undefined")
        target_compile_options(sushiblas_sanitizers INTERFACE
            "SHELL:-Xarch_host -fno-sanitize-recover=undefined")
    endif()
endif()
