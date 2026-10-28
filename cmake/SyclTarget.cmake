# SyclTarget.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SyclTarget — the SYCL policy of every target this project builds and of what the library
# publishes, in one place.
#
# On the intel-llvm toolchain a unit compiles for the device only when it instantiates a kernel:
# its target names it in DEVICE_SOURCES and it takes -fsycl and the device targets. Every other
# unit compiles for the host alone, with the toolchain's SYCL headers on its include path,
# because SushiBLAS's and SushiRuntime's public headers include <sycl/sycl.hpp>. The library
# publishes the headers and the device options as link options, so the link of an executable
# pulls the device images out of the archive, and it publishes the device compile options on
# a target of their own, SushiBLAS::device_code, for a consumer's unit that instantiates a
# SushiBLAS kernel template. docs/agent/2026_10_26_SYCL_COMPILE_TIME/SPEC.md holds the
# measurements behind this split.
#
# On the other toolchains add_sycl_to_target() is the toolchain's own function and applies to
# whole targets, and every unit of a target given to sushiblas_apply_sycl() compiles for the
# device.

include(SushiBLASSyclHeaders)

# Which devices to generate code for.
#
# Empty by default, which means the targets follow SushiRuntime: whatever its
# GPU detection resolved for this build (SushiRuntime::gpu_offload) is applied
# here too, so the library and the runtime it links cannot disagree about which
# devices carry an image. When the runtime publishes no offload target the
# build is SPIR-V only: the device image is portable intermediate code that the
# runtime JITs for whatever it finds. That reaches every OpenCL and Level Zero
# device, and reaches an NVIDIA GPU not at all - NVIDIA's OpenCL driver has
# never implemented clCreateProgramWithIL, so the CUDA image has to be compiled
# in ahead of time. A non-empty value overrides the runtime's choice.
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
    "Semicolon-separated SYCL AOT targets (e.g. spir64;nvidia_gpu_sm_86) that override SushiRuntime's. Empty means follow the runtime.")

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
    message(STATUS "SushiBLAS: AOT device targets = ${SB_SYCL_TARGETS} (SB_SYCL_TARGETS override)")
elseif(TARGET SushiRuntime::gpu_offload)
    get_target_property(_sb_runtime_targets SushiRuntime::gpu_offload INTERFACE_COMPILE_OPTIONS)
    if(_sb_runtime_targets)
        message(STATUS "SushiBLAS: device targets follow SushiRuntime = ${_sb_runtime_targets}")
    else()
        message(STATUS "SushiBLAS: SushiRuntime publishes no offload target - SPIR-V/JIT only")
    endif()
else()
    message(STATUS "SushiBLAS: no AOT device targets - SPIR-V/JIT only "
                   "(will not reach an NVIDIA GPU; set SB_SYCL_TARGETS)")
endif()

# SB_SYCL_INCLUDE_DIR: where <sycl/sycl.hpp> lives, for the units compiled without -fsycl.
# Found beside the compiler; set it by hand for a toolchain laid out otherwise.
if(SushiRuntime_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
    sushiblas_find_sycl_include_dir(_sb_sycl_include_dir)
    set(SB_SYCL_INCLUDE_DIR "${_sb_sycl_include_dir}" CACHE PATH
        "The SYCL toolchain's include directory, for units compiled without -fsycl.")
    if(NOT EXISTS "${SB_SYCL_INCLUDE_DIR}/sycl/sycl.hpp")
        message(FATAL_ERROR
            "SushiBLAS: no sycl/sycl.hpp under SB_SYCL_INCLUDE_DIR='${SB_SYCL_INCLUDE_DIR}'. "
            "Set it to the include directory of the toolchain of ${CMAKE_CXX_COMPILER}.")
    endif()
endif()

# sushiblas_runtime_device_targets(<out_compile> <out_link>)
#   Returns the device target flags SushiRuntime resolved for this build
#   (-fsycl-targets and its backend options), read from its
#   SushiRuntime::gpu_offload target, or empty lists when it publishes none.
function(sushiblas_runtime_device_targets out_compile out_link)
    set(_compile "")
    set(_link "")
    if(TARGET SushiRuntime::gpu_offload)
        get_target_property(_compile SushiRuntime::gpu_offload INTERFACE_COMPILE_OPTIONS)
        get_target_property(_link SushiRuntime::gpu_offload INTERFACE_LINK_OPTIONS)
        if(NOT _compile)
            set(_compile "")
        endif()
        if(NOT _link)
            set(_link "")
        endif()
    endif()
    set(${out_compile} "${_compile}" PARENT_SCOPE)
    set(${out_link} "${_link}" PARENT_SCOPE)
endfunction()

# sushiblas_sycl_device_options(<out_compile> <out_link>)
#   Returns the options a device unit compiles with and a SYCL executable links with on the
#   intel-llvm toolchain: -fsycl, the device targets of SB_SYCL_TARGETS or else of SushiRuntime,
#   the device optimisation level, and SB_SYCL_DEVICE_FLAGS. -fsycl-targets takes one
#   comma-separated argument, so the CMake list in SB_SYCL_TARGETS is joined with commas.
#
#   The device optimisation level is -O3 in every build type. A Debug build compiles host code at
#   -O0, and the device passes and the device link would follow it; the kernels the tests run
#   would then be too slow to test. -Xarch_device reaches the frontend of every device target;
#   the per-triple -Xsycl-target-frontend spelling reaches it too, but the driver reports it
#   unused, which -Werror turns into a failure. On the link, -O3 reaches only the device link,
#   since the host objects are already compiled.
function(sushiblas_sycl_device_options out_compile out_link)
    set(compile -fsycl)
    set(link -fsycl)
    if(SB_SYCL_TARGETS)
        string(REPLACE ";" "," target_list "${SB_SYCL_TARGETS}")
        list(APPEND compile -fsycl-targets=${target_list})
        list(APPEND link -fsycl-targets=${target_list})
    else()
        sushiblas_runtime_device_targets(runtime_compile runtime_link)
        list(APPEND compile ${runtime_compile})
        list(APPEND link ${runtime_link})
    endif()
    list(APPEND compile "SHELL:-Xarch_device -O3")
    list(APPEND link -O3)
    list(APPEND compile ${SB_SYCL_DEVICE_FLAGS})
    list(APPEND link ${SB_SYCL_DEVICE_FLAGS})
    set(${out_compile} "${compile}" PARENT_SCOPE)
    set(${out_link} "${link}" PARENT_SCOPE)
endfunction()

# SushiBLAS::sycl_headers: the SYCL headers for a unit compiled without -fsycl, quiet. The
# include directory is a path of this machine, so the export carries the definition alone and
# the installed SushiBLASConfig.cmake adds the directory it finds beside the consumer's compiler.
# SushiBLAS::device_code: the compile options of a unit that instantiates a kernel.
if(NOT TARGET sushiblas_sycl_headers)
    add_library(sushiblas_sycl_headers INTERFACE)
    add_library(SushiBLAS::sycl_headers ALIAS sushiblas_sycl_headers)
    set_target_properties(sushiblas_sycl_headers PROPERTIES EXPORT_NAME sycl_headers)

    add_library(sushiblas_device_code INTERFACE)
    add_library(SushiBLAS::device_code ALIAS sushiblas_device_code)
    set_target_properties(sushiblas_device_code PROPERTIES EXPORT_NAME device_code)

    if(SushiRuntime_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
        target_include_directories(sushiblas_sycl_headers SYSTEM INTERFACE
            "$<BUILD_INTERFACE:${SB_SYCL_INCLUDE_DIR}>")
        target_compile_definitions(sushiblas_sycl_headers INTERFACE
            SYCL_DISABLE_FSYCL_SYCLHPP_WARNING)
        sushiblas_sycl_device_options(_sb_device_compile _sb_device_link)
        target_compile_options(sushiblas_device_code INTERFACE ${_sb_device_compile})
    endif()
endif()

# sushiblas_apply_sycl(<target> [DEVICE_SOURCES <src>...])
#   Applies the SYCL policy to a target this project compiles. On intel-llvm every unit of
#   <target> gets the SYCL headers, and the DEVICE_SOURCES, which must be sources of <target>,
#   compile with the options of SushiBLAS::device_code; this function owns their
#   COMPILE_OPTIONS source property. Elsewhere add_sycl_to_target() takes every source of
#   <target>.
function(sushiblas_apply_sycl target)
    cmake_parse_arguments(ARG "" "" "DEVICE_SOURCES" ${ARGN})
    if(NOT "${ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR
            "sushiblas_apply_sycl(${target}): unexpected argument(s) '${ARG_UNPARSED_ARGUMENTS}'.")
    endif()

    if(NOT SushiRuntime_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
        get_target_property(target_sources ${target} SOURCES)
        add_sycl_to_target(TARGET ${target} SOURCES ${target_sources})
        return()
    endif()

    target_link_libraries(${target} PRIVATE sushiblas_sycl_headers)
    if(ARG_DEVICE_SOURCES)
        # A source's COMPILE_OPTIONS does not expand SHELL: groups, so each group becomes its
        # arguments here.
        get_target_property(device_compile sushiblas_device_code INTERFACE_COMPILE_OPTIONS)
        set(source_options "")
        foreach(option IN LISTS device_compile)
            if(option MATCHES "^SHELL:(.*)$")
                separate_arguments(group NATIVE_COMMAND "${CMAKE_MATCH_1}")
                list(APPEND source_options ${group})
            else()
                list(APPEND source_options "${option}")
            endif()
        endforeach()
        set_property(SOURCE ${ARG_DEVICE_SOURCES} TARGET_DIRECTORY ${target}
            PROPERTY COMPILE_OPTIONS "${source_options}")
    endif()
endfunction()

# sushiblas_publish_sycl(<library>)
#   Applies the SYCL policy to what <library> hands the targets that link it. On intel-llvm
#   they get the SYCL headers and the device link options, which pull the device images out of
#   the archive, and no device compile option. Elsewhere add_sycl_to_target() takes the whole
#   library.
function(sushiblas_publish_sycl library)
    if(NOT SushiRuntime_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
        add_sycl_to_target(TARGET ${library})
        return()
    endif()

    target_link_libraries(${library} PUBLIC sushiblas_sycl_headers)
    sushiblas_sycl_device_options(device_compile device_link)
    target_link_options(${library} INTERFACE ${device_link})
endfunction()
