# Toolchain — pick the default SYCL compiler before project()'s compiler probe.
#
# Mirror the runtime's intel-llvm default: use clang++ when the caller named no
# compiler, so a plain configure lands on the primary vendor-neutral toolchain.
# The runtime, pulled in as a subproject, does the full toolchain selection; this
# only fixes the compiler the top-level probe validates. Pass
# -DCMAKE_CXX_COMPILER=... to override (e.g. acpp, icx-cl).
if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
    find_program(SB_CLANGXX NAMES clang++)
    if(SB_CLANGXX)
        set(CMAKE_CXX_COMPILER "${SB_CLANGXX}" CACHE FILEPATH "CXX compiler")
    endif()
endif()
