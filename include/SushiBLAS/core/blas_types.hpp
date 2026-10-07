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
         * @see include/SushiBLAS/core/README.md
         */
        enum class Transpose : uint8_t
        {
            NO_TRANS = 111,   /**< op(X) = X. */
            TRANS = 112,      /**< op(X) = X^T. */
            CONJ_TRANS = 113  /**< op(X) = X^H, which is X^T for a real element type. */
        };

        /**
         * @brief Selects the triangle of a symmetric, Hermitian or triangular matrix.
         * @see include/SushiBLAS/core/README.md
         */
        enum class Uplo : uint8_t
        {
            UPPER = 121, /**< The triangle on and above the diagonal. */
            LOWER = 122  /**< The triangle on and below the diagonal. */
        };

        /**
         * @brief Selects whether a triangular matrix has a unit diagonal.
         * @see include/SushiBLAS/core/README.md
         */
        enum class Diag : uint8_t
        {
            NON_UNIT = 131, /**< The diagonal is read from the matrix. */
            UNIT = 132      /**< The diagonal is taken as all ones and never read. */
        };

        /**
         * @brief Selects the side a triangular matrix stands on.
         * @see include/SushiBLAS/core/README.md
         */
        enum class Side : uint8_t
        {
            LEFT = 141, /**< op(A) * X. */
            RIGHT = 142 /**< X * op(A). */
        };

        /** @brief Returns whether value is an enumerator of Transpose. */
        [[nodiscard]] inline constexpr bool is_valid(Transpose value) noexcept
        {
            return value == Transpose::NO_TRANS || value == Transpose::TRANS ||
                   value == Transpose::CONJ_TRANS;
        }

        /** @brief Returns whether value is an enumerator of Uplo. */
        [[nodiscard]] inline constexpr bool is_valid(Uplo value) noexcept
        {
            return value == Uplo::UPPER || value == Uplo::LOWER;
        }

        /** @brief Returns whether value is an enumerator of Diag. */
        [[nodiscard]] inline constexpr bool is_valid(Diag value) noexcept
        {
            return value == Diag::NON_UNIT || value == Diag::UNIT;
        }

        /** @brief Returns whether value is an enumerator of Side. */
        [[nodiscard]] inline constexpr bool is_valid(Side value) noexcept
        {
            return value == Side::LEFT || value == Side::RIGHT;
        }

        /** @brief Returns whether op(X) swaps the rows and columns of X. */
        [[nodiscard]] inline constexpr bool is_transposed(Transpose value) noexcept
        {
            return value != Transpose::NO_TRANS;
        }

        /** @brief Returns whether op(X) conjugates the elements of a complex X. */
        [[nodiscard]] inline constexpr bool is_conjugated(Transpose value) noexcept
        {
            return value == Transpose::CONJ_TRANS;
        }

        /** @brief Returns the name of a Transpose, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Transpose value) noexcept
        {
            switch (value)
            {
                case Transpose::NO_TRANS:
                    return "NO_TRANS";
                case Transpose::TRANS:
                    return "TRANS";
                case Transpose::CONJ_TRANS:
                    return "CONJ_TRANS";
            }
            return "UNKNOWN";
        }

        /** @brief Returns the name of a Uplo, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Uplo value) noexcept
        {
            switch (value)
            {
                case Uplo::UPPER:
                    return "UPPER";
                case Uplo::LOWER:
                    return "LOWER";
            }
            return "UNKNOWN";
        }

        /** @brief Returns the name of a Diag, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Diag value) noexcept
        {
            switch (value)
            {
                case Diag::NON_UNIT:
                    return "NON_UNIT";
                case Diag::UNIT:
                    return "UNIT";
            }
            return "UNKNOWN";
        }

        /** @brief Returns the name of a Side, or "UNKNOWN" outside the enumeration. */
        [[nodiscard]] inline constexpr const char* to_string(Side value) noexcept
        {
            switch (value)
            {
                case Side::LEFT:
                    return "LEFT";
                case Side::RIGHT:
                    return "RIGHT";
            }
            return "UNKNOWN";
        }
    } // namespace Core
} // namespace SushiBLAS
