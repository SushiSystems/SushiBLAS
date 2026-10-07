# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

cmake_minimum_required(VERSION 3.20)

# --- Define SushiBLAS Library ---
set(SUSHIBLAS_SOURCES
    # Engine
    engine.cpp

    # IO & Utilities
    io.cpp

    # BLAS: Level 1
    engine/blas/level1/axpy.cpp
    engine/blas/level1/asum.cpp
    engine/blas/level1/copy.cpp
    engine/blas/level1/dot.cpp
    engine/blas/level1/iamax.cpp
    engine/blas/level1/nrm2.cpp
    engine/blas/level1/rot.cpp
    engine/blas/level1/scal.cpp
    engine/blas/level1/swap.cpp

    # BLAS: Level 2
    engine/blas/level2/gemv.cpp
    engine/blas/level2/ger.cpp
    engine/blas/level2/symv.cpp
    engine/blas/level2/trmv.cpp
    engine/blas/level2/trsv.cpp
    engine/blas/level2/syr.cpp
    engine/blas/level2/syr2.cpp

    # BLAS: Level 3
    engine/blas/level3/gemm.cpp
    engine/blas/level3/syrk.cpp
    engine/blas/level3/trsm.cpp

    # Math: Random
    engine/math/random/bernoulli.cpp
    engine/math/random/constant.cpp
    engine/math/random/discrete_uniform.cpp
    engine/math/random/exponential.cpp
    engine/math/random/he_normal.cpp
    engine/math/random/he_uniform.cpp
    engine/math/random/log_normal.cpp
    engine/math/random/normal.cpp
    engine/math/random/orthogonal.cpp
    engine/math/random/poisson.cpp
    engine/math/random/set_seed.cpp
    engine/math/random/shuffle.cpp
    engine/math/random/truncated_normal.cpp
    engine/math/random/uniform.cpp
    engine/math/random/xavier_normal.cpp
    engine/math/random/xavier_uniform.cpp

    # Math: Nonlinear
    engine/math/nonlinear/relu.cpp
    engine/math/nonlinear/leaky_relu.cpp
    engine/math/nonlinear/sigmoid.cpp
    engine/math/nonlinear/tanh.cpp
    engine/math/nonlinear/elu.cpp
    engine/math/nonlinear/silu.cpp
    engine/math/nonlinear/gelu.cpp
    engine/math/nonlinear/softplus.cpp

    # Math: Elementwise
    engine/math/elementwise/add.cpp
    engine/math/elementwise/sub.cpp
    engine/math/elementwise/mul.cpp
    engine/math/elementwise/div.cpp
    engine/math/elementwise/sqrt.cpp
    engine/math/elementwise/square.cpp
    engine/math/elementwise/reciprocal.cpp
    engine/math/elementwise/neg.cpp
    engine/math/elementwise/exp.cpp
    engine/math/elementwise/log.cpp
    engine/math/elementwise/abs.cpp
    engine/math/elementwise/pow.cpp
    engine/math/elementwise/sin.cpp
    engine/math/elementwise/cos.cpp
    engine/math/elementwise/tan.cpp
    engine/math/elementwise/asin.cpp
    engine/math/elementwise/acos.cpp
    engine/math/elementwise/atan.cpp
    engine/math/elementwise/sinh.cpp
    engine/math/elementwise/cosh.cpp
    engine/math/elementwise/asinh.cpp
    engine/math/elementwise/acosh.cpp
    engine/math/elementwise/atanh.cpp
    engine/math/elementwise/ceil.cpp
    engine/math/elementwise/floor.cpp
    engine/math/elementwise/round.cpp
    engine/math/elementwise/clamp.cpp
    engine/math/elementwise/min.cpp
    engine/math/elementwise/max.cpp
    engine/math/elementwise/fmod.cpp
    engine/math/elementwise/remainder.cpp

    # The one op whose operands may disagree on dtype (see elementwise.hpp).
    engine/math/elementwise/convert.cpp

    # Math: Reductions
    engine/math/reductions/sum.cpp
    engine/math/reductions/mean.cpp
    engine/math/reductions/max.cpp
    engine/math/reductions/min.cpp
    engine/math/reductions/argmax.cpp
    engine/math/reductions/argmin.cpp
    engine/math/reductions/var.cpp
    engine/math/reductions/std.cpp
    engine/math/reductions/prod.cpp
    engine/math/reductions/logsumexp.cpp
    engine/math/reductions/asum.cpp
    engine/math/reductions/nrm2.cpp

    # The one reduction whose result dtype is not its input's (see reductions.hpp).
    engine/math/reductions/count_nonfinite.cpp

    # Math: Reductions along one axis (segmented; result is a vector)
    engine/math/reductions/max_along.cpp
    engine/math/reductions/min_along.cpp
    engine/math/reductions/sum_along.cpp

    # Spatial: windowed and strided movement over an N-D grid
    engine/spatial/im2col.cpp
    engine/spatial/col2im.cpp
    engine/spatial/pool_max.cpp
    engine/spatial/pool_max_backward.cpp
    engine/spatial/pool_avg.cpp
    engine/spatial/upsample_nearest.cpp
    engine/spatial/channels.cpp
    engine/spatial/broadcast.cpp

    # The one reduction whose segments are identified by a pair (see reduce_axis.hpp).
    engine/math/reductions/mean_spatial.cpp

    # Logic
    engine/logic/equal.cpp
    engine/logic/not_equal.cpp
    engine/logic/greater.cpp
    engine/logic/less.cpp
    engine/logic/greater_equal.cpp
    engine/logic/less_equal.cpp
    engine/logic/logical_and.cpp
    engine/logic/logical_or.cpp
    engine/logic/logical_xor.cpp
    engine/logic/logical_not.cpp
    engine/logic/where.cpp
    engine/logic/all.cpp
    engine/logic/any.cpp
)

add_library(sushiblas STATIC ${SUSHIBLAS_SOURCES})

# A namespaced alias so an in-tree consumer (add_subdirectory, D3's
# TARGET-guard/sibling modes) and an out-of-tree one (find_package) spell the
# dependency identically - mirrors SushiRuntime::SushiRuntime exactly (SB-6).
add_library(SushiBLAS::sushiblas ALIAS sushiblas)

# --- Include Directories ---
# Two different visibilities, not two equivalent PUBLIC paths: include/ is the
# public API surface consumers compile against (BUILD_INTERFACE while
# building this tree, INSTALL_INTERFACE afterwards - a bare absolute path
# would bake this machine's source tree into the exported target). The repo
# root is PRIVATE-only: it exists so this library's own .cpp files can
# `#include <kernels/...>`, but no public header under include/SushiBLAS/
# reaches into kernels/ - it is an implementation detail, not part of what a
# consumer links against, so it is never installed and never PUBLIC.
target_include_directories(sushiblas PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_include_directories(sushiblas PRIVATE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)

# --- Compile Policy ---
# Warnings, architecture targeting and sanitizers, PRIVATE to this tree (SB-6).
sushiblas_apply_build_policy(sushiblas)

# The compiled-in log level, PUBLIC (see cmake/SushiBLASLogging.cmake) so examples,
# tests, and out-of-tree consumers all see the same ACTIVE_LOG_LEVEL this archive was
# compiled with.
sushiblas_apply_log_level(sushiblas)

# --- Link Libraries ---
# Link against dependencies. PUBLIC so consumers (like the app) inherit them.
# SushiRuntime is always linked via its namespaced alias (D3): whichever of the
# three resolution modes in cmake/Runtime.cmake fired, the link line here is
# identical, and it never varies with how the dependency was actually found.
# This is also how SushiBLAS inherits SushiRuntime's `sushiruntime_deterministic_fp`
# contract transitively - SushiBLAS does not need (or own) a copy of that
# guarantee, since it links SushiRuntime::SushiRuntime PUBLIC.
# Not linked directly: IntelSYCL::SYCL_CXX. It's an imported target defined
# inside sushiruntime's own directory scope (non-GLOBAL, per its Sycl.cmake),
# invisible by name to sushiblas's sibling scope - but SushiRuntime::SushiRuntime
# already carries it PUBLIC, so it reaches sushiblas transitively regardless.
target_link_libraries(sushiblas PUBLIC
    SushiRuntime::SushiRuntime
)

# --- Properties ---
# C++17, matching SushiRuntime and the root CMakeLists.txt; see the note there for why
# the two must agree rather than merely both being modern.
set_target_properties(sushiblas PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# SYCL device-compilation policy (see cmake/SyclTarget.cmake) - PUBLIC, so
# examples/tests inherit it from linking sushiblas rather than reapplying it
# themselves. No oneMKL coupling: gemm/syrk are SushiBLAS's own ported/composed
# tiling kernels now (SB-2, see NOTICE.md), not an oneMKL passthrough.
sushiblas_apply_sycl(sushiblas SOURCES ${SUSHIBLAS_SOURCES})

# --- Install & export (SB-6) ---
#
# What makes SushiBLAS consumable out of tree via find_package(SushiBLAS).
# Mirrors SushiRuntime's own install/export shape exactly. Everything the
# build applies beyond this - warnings, -Werror, sanitizers, -march - is
# PRIVATE and never reaches here, which is the boundary cmake/SushiBLASBuildPolicy.cmake
# exists to draw.
install(TARGETS sushiblas
    EXPORT SushiBLASTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT SushiBLASTargets
    FILE SushiBLASTargets.cmake
    NAMESPACE SushiBLAS::
    DESTINATION "${SB_INSTALL_CMAKEDIR}"
)

# The same targets, usable from a build tree without installing. A consumer
# can point CMAKE_PREFIX_PATH at the build directory during development.
export(EXPORT SushiBLASTargets
    FILE "${PROJECT_BINARY_DIR}/SushiBLASTargets.cmake"
    NAMESPACE SushiBLAS::
)

