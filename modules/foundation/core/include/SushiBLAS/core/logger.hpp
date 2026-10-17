/**************************************************************************/
/* logger.hpp                                                             */
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
 * @file logger.hpp
 * @brief Declares the SB_LOG macros on SushiLog, SB_ASSERT and SB_THROW_IF.
 * @author Mustafa Garip
 * @see modules/foundation/core/README.md
 */

#pragma once

#include <cstdlib>
#include <stdexcept>

#include <SushiLog/log/log_front.hpp>
#include <SushiLog/log/log_level.hpp>
#include <SushiRuntime/core/text_format.hpp>


#ifndef SB_ENABLE_LOGGING
    #define SB_ENABLE_LOGGING 1
#endif

namespace SushiBLAS
{
    namespace Core
    {
        /** @brief Holds the most detailed level this translation unit compiles SB_LOG calls for. */
        constexpr SushiLog::LogLevel K_LOG_LEVEL_CEILING =
            SushiLog::log_level_ceiling(SB_ENABLE_LOGGING != 0);
    } // namespace Core
} // namespace SushiBLAS

/** @brief Logs one record at @p level in @p category through SushiLog. */
#define SB_LOG_AT(level, category, ...)                                                              SL_LOG_FRONT_AT(SushiBLAS::Core::K_LOG_LEVEL_CEILING, level, category, __VA_ARGS__)

/** @brief Logs step-by-step flow in @p category. */
#define SB_LOG_TRACE(category, ...) SB_LOG_AT(SushiLog::LogLevel::Trace, category, __VA_ARGS__)

/** @brief Logs internal state at a decision point in @p category. */
#define SB_LOG_DEBUG(category, ...) SB_LOG_AT(SushiLog::LogLevel::Debug, category, __VA_ARGS__)

/** @brief Logs a lifecycle event in @p category. */
#define SB_LOG_INFO(category, ...) SB_LOG_AT(SushiLog::LogLevel::Info, category, __VA_ARGS__)

/** @brief Logs a recovery or a fallback in @p category. */
#define SB_LOG_WARN(category, ...) SB_LOG_AT(SushiLog::LogLevel::Warn, category, __VA_ARGS__)

/** @brief Logs a failed operation in @p category. */
#define SB_LOG_ERROR(category, ...) SB_LOG_AT(SushiLog::LogLevel::Error, category, __VA_ARGS__)

/** @brief Logs, in @p category, why the process cannot continue. */
#define SB_LOG_FATAL(category, ...) SB_LOG_AT(SushiLog::LogLevel::Fatal, category, __VA_ARGS__)

/**
 * @brief Aborts with a FATAL record in @p category naming @p msg when @p condition is false;
 *        nothing under NDEBUG.
 */
#ifdef NDEBUG
    #define SB_ASSERT(category, condition, msg) ((void)0)
#else
    #define SB_ASSERT(category, condition, msg)                                                          do                                                                                               {                                                                                                    if (!(condition))                                                                                {                                                                                                    SB_LOG_FATAL(category, "assertion '{}' failed: {}; aborting", #condition, msg);                  std::abort();                                                                                }                                                                                            } while (0)
#endif

/**
 * @brief Throws std::runtime_error with the formatted message when @p condition holds; the code
 *        that catches it logs it where it handles it.
 * @param condition The condition to check.
 * @param ... The message's format string followed by its arguments.
 */
#define SB_THROW_IF(condition, ...)                                                                  do                                                                                               {                                                                                                    if (condition)                                                                                       throw std::runtime_error(SushiRuntime::format_text(__VA_ARGS__));                  } while (0)
