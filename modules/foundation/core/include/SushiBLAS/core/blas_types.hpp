/**************************************************************************/
/* blas_types.hpp                                                         */
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
 * @file blas_types.hpp
 * @brief Declares the flags of a BLAS call: Transpose, Uplo, Diag and Side.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiBLAS
{
    namespace Core
    {
        /**
         * @brief Selects op(X) for a matrix operand.
         * @see modules/foundation/core/README.md
         */
        enum class Transpose : uint8_t
        {
            NoTrans = 111,   /**< op(X) = X. */
            Trans = 112,      /**< op(X) = X^T. */
            ConjTrans = 113  /**< op(X) = X^H, which is X^T for a real element type. */
        };

        /**
         * @brief Selects the triangle of a symmetric, Hermitian or triangular matrix.
         * @see modules/foundation/core/README.md
         */
        enum class Uplo : uint8_t
        {
            Upper = 121, /**< The triangle on and above the diagonal. */
            Lower = 122  /**< The triangle on and below the diagonal. */
        };

        /**
         * @brief Selects whether a triangular matrix has a unit diagonal.
         * @see modules/foundation/core/README.md
         */
        enum class Diag : uint8_t
        {
            NonUnit = 131, /**< The diagonal is read from the matrix. */
            Unit = 132      /**< The diagonal is taken as all ones and never read. */
        };

        /**
         * @brief Selects the side a triangular matrix stands on.
         * @see modules/foundation/core/README.md
         */
        enum class Side : uint8_t
        {
            Left = 141, /**< op(A) * X. */
            Right = 142 /**< X * op(A). */
        };

        /** @brief Returns whether value is an enumerator of Transpose. */
        [[nodiscard]] inline constexpr bool is_valid(Transpose value) noexcept
        {
            return value == Transpose::NoTrans || value == Transpose::Trans ||
                   value == Transpose::ConjTrans;
        }

        /** @brief Returns whether value is an enumerator of Uplo. */
        [[nodiscard]] inline constexpr bool is_valid(Uplo value) noexcept
        {
            return value == Uplo::Upper || value == Uplo::Lower;
        }

        /** @brief Returns whether value is an enumerator of Diag. */
        [[nodiscard]] inline constexpr bool is_valid(Diag value) noexcept
        {
            return value == Diag::NonUnit || value == Diag::Unit;
        }

        /** @brief Returns whether value is an enumerator of Side. */
        [[nodiscard]] inline constexpr bool is_valid(Side value) noexcept
        {
            return value == Side::Left || value == Side::Right;
        }

        /** @brief Returns whether op(X) swaps the rows and columns of X. */
        [[nodiscard]] inline constexpr bool is_transposed(Transpose value) noexcept
        {
            return value != Transpose::NoTrans;
        }

        /** @brief Returns whether op(X) conjugates the elements of a complex X. */
        [[nodiscard]] inline constexpr bool is_conjugated(Transpose value) noexcept
        {
            return value == Transpose::ConjTrans;
        }

        /** @brief Returns the name of a Transpose, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Transpose value) noexcept
        {
            switch (value)
            {
                case Transpose::NoTrans:
                    return "NO_TRANS";
                case Transpose::Trans:
                    return "TRANS";
                case Transpose::ConjTrans:
                    return "CONJ_TRANS";
            }
            return "UNKNOWN";
        }

        /** @brief Returns the name of a Uplo, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Uplo value) noexcept
        {
            switch (value)
            {
                case Uplo::Upper:
                    return "UPPER";
                case Uplo::Lower:
                    return "LOWER";
            }
            return "UNKNOWN";
        }

        /** @brief Returns the name of a Diag, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Diag value) noexcept
        {
            switch (value)
            {
                case Diag::NonUnit:
                    return "NON_UNIT";
                case Diag::Unit:
                    return "UNIT";
            }
            return "UNKNOWN";
        }

        /** @brief Returns the name of a Side, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Side value) noexcept
        {
            switch (value)
            {
                case Side::Left:
                    return "LEFT";
                case Side::Right:
                    return "RIGHT";
            }
            return "UNKNOWN";
        }
    } // namespace Core
} // namespace SushiBLAS
