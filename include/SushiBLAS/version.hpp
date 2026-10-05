/****************************************************************/
/* version.hpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file version.hpp
 * @brief The library's version.
 *
 * Unlike SushiRuntime (a SHARED library, where "the headers compiled
 * against" and "the .dll actually loaded" can legitimately disagree),
 * SushiBLAS is STATIC: there is no separate loaded artifact whose version
 * could drift from these macros at runtime, so a single compile-time version
 * is the whole story here.
 *
 * The literals here are checked against the project's CMake version at
 * configure time (see the root CMakeLists.txt), so this header and the build
 * cannot drift apart.
 */

/** @brief Major version. */
#define SUSHIBLAS_VERSION_MAJOR 0
/** @brief Minor version. */
#define SUSHIBLAS_VERSION_MINOR 1
/** @brief Patch version. */
#define SUSHIBLAS_VERSION_PATCH 0

/** @brief Dotted version, as a literal. */
#define SUSHIBLAS_VERSION_STRING "0.1.0"

/**
 * @brief The version as one comparable integer.
 *
 * Encoded so ordinary integer comparison orders releases:
 * `#if SUSHIBLAS_VERSION >= SUSHIBLAS_VERSION_ENCODE(0, 2, 0)`.
 */
#define SUSHIBLAS_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

/** @brief The version, encoded by SUSHIBLAS_VERSION_ENCODE. */
#define SUSHIBLAS_VERSION                                \
    SUSHIBLAS_VERSION_ENCODE(SUSHIBLAS_VERSION_MAJOR,    \
                             SUSHIBLAS_VERSION_MINOR,    \
                             SUSHIBLAS_VERSION_PATCH)
