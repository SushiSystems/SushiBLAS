# SushiBLASLogging.cmake — the compiled-in log level, as a build option.
#
# Single responsibility: turn one cache variable into the one compile definition
# include/SushiBLAS/core/logger.hpp reads, and expose a function that applies it.
#
# The level used to be a bare `#ifndef`/`#ifdef NDEBUG` pair inside the header, which
# meant raising it for a debugging session required editing a tracked file. It is a
# build-time choice, so it belongs in the build.
#
# Prefixed with the program's full name for the reason cmake/SushiBLASBuildPolicy.cmake
# spells out: CMAKE_MODULE_PATH is one flat namespace once sushiruntime's own cmake/ is
# added alongside this one.

# `auto` (the default) leaves the choice to the header, which picks WARN under NDEBUG
# and DEBUG otherwise. Any other value is passed straight through as an enumerator name,
# so a value the header does not recognise fails to compile rather than degrading
# silently — but the validation below catches it at configure time instead, which is a
# far better error message.
set(SB_LOG_LEVEL "auto" CACHE STRING
    "Compiled-in SushiBLAS log level: auto | OFF | ERR | WARN | INFO | DEBUG \
('auto' = WARN for release, DEBUG for debug)")

set(_sb_log_levels auto OFF ERR WARN INFO DEBUG)
set_property(CACHE SB_LOG_LEVEL PROPERTY STRINGS ${_sb_log_levels})

if(NOT SB_LOG_LEVEL IN_LIST _sb_log_levels)
    message(FATAL_ERROR
        "SB_LOG_LEVEL is '${SB_LOG_LEVEL}'. Expected one of: ${_sb_log_levels}. "
        "These are the SushiBLAS::Core::LogType enumerator names, plus 'auto' for the "
        "per-build-type default.")
endif()

# Applies the configured log level to one target.
#
# PUBLIC, unlike the warning/sanitizer policy: ACTIVE_LOG_LEVEL is an `inline constexpr`
# variable in a header every consumer compiles, so the library and everything linking it
# have to agree on its value or they are an ODR violation waiting to be noticed. The
# definition is a value, not a target name, so it travels through install(EXPORT)
# without the export-set problem sushiblas_apply_build_policy() exists to avoid.
function(sushiblas_apply_log_level target)
    if(NOT SB_LOG_LEVEL STREQUAL "auto")
        target_compile_definitions(${target} PUBLIC SB_ACTIVE_LOG_LEVEL=${SB_LOG_LEVEL})
    endif()
endfunction()
