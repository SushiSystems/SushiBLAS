/**************************************************************************/
/* band_operand.hpp                                                       */
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
 * @file band_operand.hpp
 * @brief Declares BandOperand, the geometry of one band-stored matrix operand of a BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the stored lines, the leading dimension, the band widths and the layout of
         *        one band operand.
         * @see modules/operations/blas/README.md
         */
        struct BandOperand
        {
            int64_t lines = 0;             /**< Columns (column-major) or rows (row-major). */
            int64_t leading_dimension = 0; /**< Elements per line, at least lower + upper + 1. */
            int64_t lower = 0;             /**< Sub-diagonals of the matrix, kl. */
            int64_t upper = 0;             /**< Super-diagonals of the matrix, ku. */
            Core::Layout layout = Core::Layout::RowMajor; /**< Which band storage it is. */
        };

        /**
         * @brief Returns the geometry of a band operand with kl sub- and ku super-diagonals.
         * @param kl_name Names kl in a message, such as "kl" or "k"; ku_name names ku.
         * @throws std::invalid_argument on a tensor that is no contiguous rank-2 matrix with
         *         storage, a negative band width, or fewer than kl + ku + 1 elements per line.
         * @throws UnsupportedExecution on a dtype the tensor's device cannot compute in.
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] inline BandOperand describe_band_operand(const Tensor& t, int64_t kl,
                                                               int64_t ku, const char* op,
                                                               const char* operand,
                                                               const char* kl_name = "kl",
                                                               const char* ku_name = "ku")
        {
            if (kl < 0)
                Detail::throw_operand_error(op, kl_name, "is {}; a band width cannot be negative",
                                            kl);
            if (ku < 0)
                Detail::throw_operand_error(op, ku_name, "is {}; a band width cannot be negative",
                                            ku);

            const MatrixOperand m = describe_matrix_operand(t, op, operand);
            require_unbatched(m, op, operand);

            const bool by_rows = (m.layout == Core::Layout::RowMajor);

            // Written without kl + ku, which two large widths could overflow.
            if (ku >= m.leading_dimension || kl > m.leading_dimension - 1 - ku)
                Detail::throw_operand_error(op, operand,
                                            "stores {} elements per {}; a band of {} = {} "
                                            "sub-diagonals and {} = {} super-diagonals needs "
                                            "one more than their sum",
                                            m.leading_dimension, by_rows ? "row" : "column",
                                            kl_name, kl, ku_name, ku);

            BandOperand b;
            b.lines = by_rows ? m.rows : m.cols;
            b.leading_dimension = m.leading_dimension;
            b.lower = kl;
            b.upper = ku;
            b.layout = m.layout;
            return b;
        }

        /**
         * @brief Returns the geometry of the band storage of one triangle with k off-diagonals.
         * @throws std::invalid_argument as describe_band_operand states, naming the width "k".
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] inline BandOperand describe_triangle_band_operand(const Tensor& t,
                                                                        Core::Uplo uplo, int64_t k,
                                                                        const char* op,
                                                                        const char* operand)
        {
            const bool upper = (uplo == Core::Uplo::Upper);
            return describe_band_operand(t, upper ? int64_t{0} : k, upper ? k : int64_t{0}, op,
                                         operand, "k", "k");
        }
    } // namespace Dispatch
} // namespace SushiBLAS
