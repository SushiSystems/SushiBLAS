# SyclTarget — the single source of truth for turning a target's sources into
# SYCL device code against the runtime. Unlike SushiEngine (header-only, so
# every consumer instantiates its kernels itself), SushiBLAS compiles its own
# SYCL kernels into the `sushiblas` library, so the flags below are applied
# PUBLIC: examples/tests just link `sushiblas` and inherit them, rather than
# every consumer calling this function itself.

# sushiblas_apply_sycl(<target> [SOURCES <src>...])
#   Applies the SYCL device-compilation policy to <target>:
#     * the toolchain's add_sycl_to_target() (pass SOURCES so AdaptiveCpp/oneAPI
#       can find the device code; harmless where it is a no-op stub),
#     * an explicit -fsycl on the intel-llvm path, where add_sycl_to_target() is a
#       no-op stub and the flag must be added ourselves - PUBLIC, so it reaches
#       whatever links <target> too.
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
    endif()
endfunction()
