# SushiBLASLogging.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLASLogging.cmake — the compiled-in log level, as a build option.
#
# Single responsibility: turn one cache variable into the one compile definition
# modules/foundation/core/include/SushiBLAS/core/logger.hpp reads, and expose a function that applies it.
#
# The level used to be a bare `#ifndef`/`#ifdef NDEBUG` pair inside the header, which
# meant raising it for a debugging session required editing a tracked file. It is a
# build-time choice, so it belongs in the build.
#
# Prefixed with the program's full name for the reason cmake/SushiBLASBuildPolicy.cmake
# spells out: CMAKE_MODULE_PATH is one flat namespace once sushiruntime's own cmake/ is
# added alongside this one.

# `auto` (the default) leaves the choice to the header, which picks WARN under NDEBUG
# and DEBUG otherwise. Any other value is validated here and handed to the header as the
# SushiBLAS::Core::LogType enumerator of that level.
set(SB_LOG_LEVEL "auto" CACHE STRING
    "Compiled-in SushiBLAS log level: auto | OFF | ERR | WARN | INFO | DEBUG \
('auto' = WARN for release, DEBUG for debug)")

set(_sb_log_levels auto OFF ERR WARN INFO DEBUG)
set_property(CACHE SB_LOG_LEVEL PROPERTY STRINGS ${_sb_log_levels})

if(NOT SB_LOG_LEVEL IN_LIST _sb_log_levels)
    message(FATAL_ERROR
        "SB_LOG_LEVEL is '${SB_LOG_LEVEL}'. Expected one of: ${_sb_log_levels}. "
        "'auto' is the per-build-type default.")
endif()

# The LogType enumerator each level name stands for.
set(_sb_log_enumerator_OFF Off)
set(_sb_log_enumerator_ERR Err)
set(_sb_log_enumerator_WARN Warn)
set(_sb_log_enumerator_INFO Info)
set(_sb_log_enumerator_DEBUG Debug)

# Applies the configured log level to one target. The scope is PUBLIC unless the caller
# names another, which an INTERFACE target has to.
#
# PUBLIC, unlike the warning/sanitizer policy: K_ACTIVE_LOG_LEVEL is an `inline constexpr`
# variable in a header every consumer compiles, so the library and everything linking it
# have to agree on its value or they are an ODR violation waiting to be noticed. The
# definition is a value, not a target name, so it travels through install(EXPORT)
# without the export-set problem sushiblas_apply_build_policy() exists to avoid.
function(sushiblas_apply_log_level target)
    set(scope PUBLIC)
    if(ARGC GREATER 1)
        set(scope ${ARGV1})
    endif()
    if(NOT SB_LOG_LEVEL STREQUAL "auto")
        target_compile_definitions(${target} ${scope}
            SB_ACTIVE_LOG_LEVEL=${_sb_log_enumerator_${SB_LOG_LEVEL}})
    endif()
endfunction()
