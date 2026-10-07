/**************************************************************************/
/* symmetric_access.hpp                                                   */
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
 * @file symmetric_access.hpp
 * @brief Declares SymmetricAccess, which reads a symmetric or Hermitian matrix from one triangle.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/kernels/blas/conjugation.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Reads any element of a symmetric or Hermitian matrix from its stored triangle.
         * @tparam Conj Unconjugated for a symmetric matrix, Conjugated for a Hermitian one.
         * @see include/SushiBLAS/kernels/README.md
         */
        template<typename Conj>
        struct SymmetricAccess
        {
            OperandAccess stored; /**< How the matrix is stored; never transposed. */
            bool upper = false;   /**< Whether the stored triangle is the upper one. */

            /**
             * @brief Returns element (r, c) of the full matrix from one read of the triangle.
             * @pre (r, c) lies inside the matrix.
             */
            template<typename T>
            [[nodiscard]] constexpr T at(const T* p, int64_t r, int64_t c) const noexcept
            {
                const bool mirrored = upper ? (r > c) : (r < c);
                const T value = p[mirrored ? stored.index(c, r) : stored.index(r, c)];
                if (r == c)
                    return Conj::diagonal(value);
                return mirrored ? Conj::apply(value) : value;
            }
        };

        /**
         * @brief Resolves a layout, a Uplo and a leading dimension into a SymmetricAccess.
         * @tparam Conj Unconjugated for a symmetric matrix, Conjugated for a Hermitian one.
         * @see include/SushiBLAS/kernels/README.md
         */
        template<typename Conj>
        [[nodiscard]] inline constexpr SymmetricAccess<Conj> symmetric_access(Core::Layout layout,
                                                                              Core::Uplo uplo,
                                                                              int64_t ld) noexcept
        {
            return SymmetricAccess<Conj>{operand_access(layout, Core::Transpose::NoTrans, ld),
                                         uplo == Core::Uplo::Upper};
        }
    } // namespace Kernels
} // namespace SushiBLAS
