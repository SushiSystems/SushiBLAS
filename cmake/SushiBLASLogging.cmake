# SushiBLASLogging.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASLogging.cmake: whether SushiBLAS compiles its log calls below ERROR, as a build option.
#
# Single responsibility: turn the SB_ENABLE_LOGGING option into the one compile definition
# modules/foundation/core/include/SushiBLAS/core/logger.hpp reads, and expose a function that
# applies it. The level a record must reach at run time is set by the SB_LOG_LEVEL environment
# variable, not here.
#
# Prefixed with the program's full name for the reason cmake/SushiBLASBuildPolicy.cmake
# spells out: CMAKE_MODULE_PATH is one flat namespace once sushiruntime's own cmake/ is
# added alongside this one.

option(SB_ENABLE_LOGGING
    "Compile SushiBLAS log calls below ERROR (TRACE in a debug build, DEBUG under NDEBUG)" ON)

# Applies SB_ENABLE_LOGGING to <target>, PUBLIC unless a scope is given.
function(sushiblas_apply_logging target)
    set(scope PUBLIC)
    if(ARGC GREATER 1)
        set(scope ${ARGV1})
    endif()
    if(NOT SB_ENABLE_LOGGING)
        target_compile_definitions(${target} ${scope} SB_ENABLE_LOGGING=0)
    endif()
endfunction()
