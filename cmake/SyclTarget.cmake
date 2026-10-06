# SyclTarget.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SyclTarget — the single source of truth for turning a target's sources into
# SYCL device code against the runtime. Unlike SushiEngine (header-only, so
# every consumer instantiates its kernels itself), SushiBLAS compiles its own
# SYCL kernels into the `sushiblas` library, so the flags below are applied
# PUBLIC: examples/tests just link `sushiblas` and inherit them, rather than
# every consumer calling this function itself.

# Which devices to generate code for.
#
# Empty by default, which means SPIR-V only: the device image is generated as
# portable intermediate code and the runtime JITs it for whatever it finds. That
# reaches every OpenCL and Level Zero device, and reaches an NVIDIA GPU not at
# all - NVIDIA's OpenCL driver has never implemented clCreateProgramWithIL, so
# there is no way for it to ingest SPIR-V. Running on CUDA is therefore not a
# runtime choice; the CUDA image has to be compiled in ahead of time, and that
# is what this option turns on.
#
# Setting it requires an intel-llvm built with `--cuda`; a stock oneAPI release
# has no CUDA backend and will reject the triple. Note also that intel-llvm
# documents support for sm_75 and newer, so older cards are below the tested
# floor even when the flag is accepted.
#
# Usually you want the SPIR-V image kept alongside the CUDA one, so the same
# binary still runs on the CPU device:
#
#   sb build --type release -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"
#
# Named SB_* rather than SUSHIBLAS_* to match SB_NATIVE_ARCH, the only other
# option this repo defines.
set(SB_SYCL_TARGETS "" CACHE STRING
    "Semicolon-separated SYCL AOT targets (e.g. spir64;nvidia_gpu_sm_86). Empty means SPIR-V/JIT only, which cannot reach an NVIDIA GPU.")

# Extra flags for the device compiler, appended verbatim to both compile and
# link. The escape hatch for backend-specific switches this file should not need
# to know about - most usefully `-Xcuda-ptxas;-v`, which makes ptxas report
# register and spill counts per kernel. A GEMM that spills its accumulator to
# local memory loses most of the benefit of blocking, and the spill count is the
# only way to see it happen.
set(SB_SYCL_DEVICE_FLAGS "" CACHE STRING
    "Extra flags passed to the SYCL device compiler and linker (e.g. -Xcuda-ptxas;-v to report register spills).")

# Reported once at configure time, not per target: which devices a binary can
# actually reach is the kind of thing that is painful to discover later, from a
# runtime "no device found" on a machine that plainly has one.
if(SB_SYCL_TARGETS)
    message(STATUS "SushiBLAS: AOT device targets = ${SB_SYCL_TARGETS}")
else()
    message(STATUS "SushiBLAS: no AOT device targets - SPIR-V/JIT only "
                   "(will not reach an NVIDIA GPU; set SB_SYCL_TARGETS)")
endif()

# sushiblas_apply_sycl(<target> [SOURCES <src>...])
#   Applies the SYCL device-compilation policy to <target>:
#     * the toolchain's add_sycl_to_target() (pass SOURCES so AdaptiveCpp/oneAPI
#       can find the device code; harmless where it is a no-op stub),
#     * an explicit -fsycl on the intel-llvm path, where add_sycl_to_target() is a
#       no-op stub and the flag must be added ourselves - PUBLIC, so it reaches
#       whatever links <target> too,
#     * the AOT target list, if one was requested.
#
#   The AOT flags have to be PUBLIC for the same reason -fsycl does, only more
#   so: sushiblas is a STATIC library, so its device code is not finalised until
#   the final executable links. A consumer that inherits -fsycl but not
#   -fsycl-targets links a host binary carrying no CUDA image and then reports,
#   at run time, that it cannot find a device - with nothing in the build log to
#   suggest why.
function(sushiblas_apply_sycl target)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    if(ARG_SOURCES)
        add_sycl_to_target(TARGET ${target} SOURCES ${ARG_SOURCES})
    else()
        add_sycl_to_target(TARGET ${target})
    endif()

    if(SR_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
        target_compile_options(${target} PUBLIC -fsycl)
        target_link_options(${target} PUBLIC -fsycl)

        if(SB_SYCL_TARGETS)
            # -fsycl-targets takes one comma-separated argument; CMake lists are
            # semicolon-separated, and passing one through unconverted hands the
            # driver a single unparseable triple.
            string(REPLACE ";" "," _sb_target_list "${SB_SYCL_TARGETS}")
            target_compile_options(${target} PUBLIC -fsycl-targets=${_sb_target_list})
            target_link_options(${target} PUBLIC -fsycl-targets=${_sb_target_list})
        endif()

        if(SB_SYCL_DEVICE_FLAGS)
            target_compile_options(${target} PUBLIC ${SB_SYCL_DEVICE_FLAGS})
            target_link_options(${target} PUBLIC ${SB_SYCL_DEVICE_FLAGS})
        endif()
    endif()
endfunction()
