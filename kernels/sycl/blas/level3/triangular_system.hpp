/**************************************************************************/
/* triangular_system.hpp                                                  */
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
 * @file triangular_system.hpp
 * @brief Declares TriangularSystem, the one substitution every side and flag of TRSM decodes to.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Holds a triangular solve as steps and vectors: unknown (s, v) is step s of the
             *        substitution for right-hand-side vector v.
             * @see kernels/README.md
             */
            struct TriangularSystem
            {
                OperandAccess coefficient; /**< Reads the factor of unknown t in step s. */
                OperandAccess unknown;     /**< Gives the offset of unknown (s, v) in B. */
                int64_t order = 0;         /**< Steps of one substitution: the order of A. */
                int64_t vectors = 0;       /**< Independent substitutions in one matrix B. */
                bool forward = true;       /**< Whether step s needs the steps below it. */
                bool unit_diag = false;    /**< Whether the diagonal is one and never read. */
            };

            /**
             * @brief Decodes the flags and extents of a TRSM call into a TriangularSystem.
             * @param m Rows of B.
             * @param n Columns of B.
             * @see kernels/README.md
             */
            [[nodiscard]] inline constexpr TriangularSystem triangular_system(
                Core::Layout layout, Core::Side side, Core::Uplo uplo, Core::Transpose trans,
                Core::Diag diag, int64_t m, int64_t n, int64_t lda, int64_t ldb) noexcept
            {
                const bool left = (side == Core::Side::LEFT);
                const bool op_upper = Core::is_transposed(trans) != (uplo == Core::Uplo::UPPER);
                const OperandAccess op_a = operand_access(layout, trans, lda);
                const OperandAccess at_b = operand_access(layout, Core::Transpose::NO_TRANS, ldb);

                return TriangularSystem{left ? op_a : op_a.transposed(),
                                        left ? at_b : at_b.transposed(),
                                        left ? m : n,
                                        left ? n : m,
                                        left ? !op_upper : op_upper,
                                        diag == Core::Diag::UNIT};
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
