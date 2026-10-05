/****************************************************************/
/* version.hpp                                                  */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file version.hpp
 * @brief Defines the SushiBLAS version macros.
 * @author Mustafa Garip
 */

#pragma once

/**
 * @brief Major version.
 * @see include/SushiBLAS/README.md
 */
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
