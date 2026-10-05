# SushiBLASBuildPolicy.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASBuildPolicy.cmake — the single entry point for this tree's compile policy.
#
# Single responsibility: compose the policy targets and expose one function
# that applies them to a target. Every target in the tree — library, examples,
# tests — goes through sushiblas_apply_build_policy() and therefore compiles
# under identical settings.
#
# Named with the full SushiBLAS prefix (not just "BuildPolicy.cmake" like its
# sushiruntime counterpart) because CMAKE_MODULE_PATH is a flat, shared
# namespace once D3's sibling add_subdirectory brings sushiruntime's own
# cmake/ onto the same path list: a same-named file here would shadow (or be
# shadowed by) sushiruntime's own BuildPolicy.cmake, and whichever
# `include(BuildPolicy)` call runs second would silently load the wrong
# file's functions/targets. Discovered the hard way while adding this file:
# sushiruntime's own build broke with "unknown function
# sushiruntime_apply_build_policy" the moment a same-named BuildPolicy.cmake
# existed in sushiblas/cmake/ too. The other three policy files below (and
# their include() calls) are prefixed for the same reason. Pre-existing,
# unaffected by this fix: sushiblas/cmake/Vcpkg.cmake shares its name with
# sushiruntime/cmake/Vcpkg.cmake too, and is exposed to the same shadowing
# risk — not fixed here since both files' vcpkg-detection logic happens to be
# idempotent either way, but worth a cross-repo naming pass someday.
#
# All three targets composed here (sushiblas_warnings, sushiblas_perf,
# sushiblas_sanitizers) are PRIVATE, in-tree-only policy. Unlike SushiRuntime,
# SushiBLAS does not own a floating-point contract to export: it already
# receives SushiRuntime's `sushiruntime_deterministic_fp` transitively (PUBLIC
# link on SushiRuntime::SushiRuntime, see src/CMakeLists.txt), which is the
# correct owner of that guarantee since SushiBLAS's kernels sit downstream of
# it, not beside it.
#
# Warnings are a review standard, architecture targeting is a choice about the
# machine a binary is built for, and a sanitizer is a debugging tool — none of
# the three is something an installed consumer should silently inherit.
# Applying this via a function rather than a directory-scope add_compile_options()
# is what makes that boundary enforceable and keeps the in-tree-only flags out
# of the installed export set — see sushiruntime/cmake/BuildPolicy.cmake for
# the fuller rationale this mirrors.

include(SushiBLASCompilerWarnings)  # -> sushiblas_warnings
include(SushiBLASOptimization)      # -> sushiblas_perf
include(SushiBLASSanitizers)        # -> sushiblas_sanitizers

# Applies the project's in-tree compile policy to one target.
#
# Deliberately NOT `target_link_libraries(target PRIVATE sushiblas_warnings)`,
# even BUILD_INTERFACE-wrapped, the way sushiruntime's own BuildPolicy.cmake
# does it. That pattern relies on PRIVATE dependencies of a library never
# reaching its exported interface - true for a SHARED library (sushiruntime),
# false for a STATIC one (sushiblas): CMake must record a static library's
# PRIVATE link dependencies in its own INTERFACE_LINK_LIBRARIES (as
# $<LINK_ONLY:...>), because a consumer linking a static archive has to link
# its dependencies too. BUILD_INTERFACE only controls WHEN that reference is
# active, not WHETHER it's recorded - so `export(EXPORT ...)` (which writes a
# build-tree package where BUILD_INTERFACE genuinely resolves true) still
# needs sushiblas_warnings to be part of an export set, and errors with
# "requires target ... that is not in any export set" otherwise. Discovered
# by actually running the export step (not `-fsyntax-only` confidence, see
# the runtime's own P6) rather than assumed from the sushiruntime precedent.
#
# The fix: pull the *values* off the policy targets' INTERFACE_* properties
# with target_compile_options()/target_compile_definitions()/target_link_options()
# instead of linking the targets themselves. Those three (unlike
# target_link_libraries) never record the source target's name anywhere - only
# the resulting flag list - so there is no dependency for export() to trip on,
# and no BUILD_INTERFACE wrapper is needed at all: PRIVATE compile/link options
# never enter a target's own exported interface, static or shared.
function(sushiblas_apply_build_policy target)
    target_compile_options(${target} PRIVATE
        "$<TARGET_PROPERTY:sushiblas_warnings,INTERFACE_COMPILE_OPTIONS>"
        "$<TARGET_PROPERTY:sushiblas_perf,INTERFACE_COMPILE_OPTIONS>"
        "$<TARGET_PROPERTY:sushiblas_sanitizers,INTERFACE_COMPILE_OPTIONS>"
    )
    target_compile_definitions(${target} PRIVATE
        "$<TARGET_PROPERTY:sushiblas_warnings,INTERFACE_COMPILE_DEFINITIONS>"
    )
    target_link_options(${target} PRIVATE
        "$<TARGET_PROPERTY:sushiblas_sanitizers,INTERFACE_LINK_OPTIONS>"
    )
endfunction()
