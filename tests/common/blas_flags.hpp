/**************************************************************************/
/* blas_flags.hpp                                                         */
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
 * @file blas_flags.hpp
 * @brief Declares the mappings from a test's bool parameters to the BLAS flag enumerations.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/core/blas_types.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns TRANS for true and NO_TRANS for false. */
        [[nodiscard]] constexpr Core::Transpose transpose_of(bool transposed) noexcept
        {
            return transposed ? Core::Transpose::TRANS : Core::Transpose::NO_TRANS;
        }

        /** @brief Returns UPPER for true and LOWER for false. */
        [[nodiscard]] constexpr Core::Uplo uplo_of(bool upper) noexcept
        {
            return upper ? Core::Uplo::UPPER : Core::Uplo::LOWER;
        }

        /** @brief Returns UNIT for true and NON_UNIT for false. */
        [[nodiscard]] constexpr Core::Diag diag_of(bool unit) noexcept
        {
            return unit ? Core::Diag::UNIT : Core::Diag::NON_UNIT;
        }

        /** @brief Returns LEFT for true and RIGHT for false. */
        [[nodiscard]] constexpr Core::Side side_of(bool left) noexcept
        {
            return left ? Core::Side::LEFT : Core::Side::RIGHT;
        }
    } // namespace Test
} // namespace SushiBLAS
