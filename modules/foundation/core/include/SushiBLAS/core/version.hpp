/**************************************************************************/
/* version.hpp                                                            */
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
 * @file version.hpp
 * @brief Defines the SushiBLAS version macros.
 * @author Mustafa Garip
 */

#pragma once

/**
 * @brief Major version.
 * @see modules/README.md
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
