/**************************************************************************/
/* profile.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file profile.hpp
 * @brief Declares the SB_ZONE family, SushiBLAS's front over SushiProfiler.
 * @author Mustafa Garip
 * @see modules/foundation/profile/README.md
 */

#pragma once

#include <mutex>
#include <type_traits>
#include <SushiBLAS/profile/profile_config.hpp>
#include <SushiBLAS/profile/profile_gate.hpp>
#include <SushiBLAS/profile/profile_keys.hpp>

#if SB_ENABLE_PROFILER

#include <SushiProfiler/lock_wait/contended_lock.hpp>
#include <SushiProfiler/zones/macros.hpp>

/** @brief Records the rest of the scope as a stage zone, level 1. */
#define SB_ZONE(name)                                                                              \
    SP_ZONE_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, 1, name,                          \
               ::SushiBLAS::Internal::Profile::K_GROUP_NAME.data())

/** @brief Records the rest of the scope as a fine zone, level 2. */
#define SB_ZONE_FINE(name)                                                                         \
    SP_ZONE_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, 2, name,                          \
               ::SushiBLAS::Internal::Profile::K_GROUP_NAME.data())

/** @brief Records the rest of the scope as a per-element zone, level 3. */
#define SB_ZONE_DETAIL(name)                                                                       \
    SP_ZONE_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, 3, name,                          \
               ::SushiBLAS::Internal::Profile::K_GROUP_NAME.data())

/** @brief Records @p value as a sample of @p unit, level 1. */
#define SB_VALUE(name, unit, value)                                                                \
    SP_VALUE_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, 1, name, unit, value)

/** @brief Records @p value at a counter folded per period by CounterOp @p op, level 1. */
#define SB_COUNTER(name, unit, op, value)                                                          \
    SP_COUNTER_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, 1, name, unit, op, value)

/** @brief Records an occurrence now, level 1. */
#define SB_INSTANT(name)                                                                           \
    SP_INSTANT_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, 1, name)

/** @brief Locks @p mutex for the rest of the scope and records its wait when it was contended. */
#define SB_LOCK_WAIT(name, mutex)                                                                  \
    SP_LOCK_WAIT_AT(::SushiBLAS::Internal::Profile::K_COMPILED_LEVEL, name, mutex)

#else // SB_ENABLE_PROFILER

#define SB_ZONE(name) static_cast<void>(0)
#define SB_ZONE_FINE(name) static_cast<void>(0)
#define SB_ZONE_DETAIL(name) static_cast<void>(0)
/** @brief Evaluates nothing: the unevaluated operand keeps a variable only a value reads from being reported unused. */
#define SB_VALUE(name, unit, value) static_cast<void>(sizeof(value))
#define SB_COUNTER(name, unit, op, value) static_cast<void>(sizeof(value))
#define SB_INSTANT(name) static_cast<void>(0)

/** @brief Locks @p mutex for the rest of the scope; with no profiler the lock is a plain one. */
#define SB_LOCK_WAIT(name, mutex)                                                                  \
    const std::lock_guard<std::remove_reference_t<decltype(mutex)>> sb_lock_wait_guard_##__LINE__(mutex)

#endif // SB_ENABLE_PROFILER
