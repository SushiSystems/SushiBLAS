/**************************************************************************/
/* matrix_operand.hpp                                                     */
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
 * @file matrix_operand.hpp
 * @brief Declares MatrixOperand, the batch and matrix geometry of one dense matrix operand of a
 *        BLAS call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the batch count, matrix size, leading dimension and layout of one operand.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        struct MatrixOperand
        {
            int32_t rank = 0;                                /**< Rank of the tensor, at least 2. */
            int64_t batch_count = 1;                         /**< Product of every axis before the last two. */
            int64_t rows = 0;                                /**< Size of the second-to-last axis. */
            int64_t cols = 0;                                /**< Size of the last axis. */
            int64_t leading_dimension = 0;                   /**< Distance between rows (row-major) or columns (column-major). */
            int64_t matrix_elements = 0;                     /**< Elements in one matrix of the batch. */
            Core::Layout layout = Core::Layout::RowMajor;   /**< Order of the elements within one matrix. */
        };

        /**
         * @brief Returns the geometry of a dense matrix operand of rank 2 or more.
         * @throws std::invalid_argument on rank < 2, missing storage or a non-contiguous tensor.
         * @throws UnsupportedExecution on a dtype the tensor's device cannot compute in.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        [[nodiscard]] inline MatrixOperand describe_matrix_operand(const Tensor& t, const char* op,
                                                                   const char* operand)
        {
            if (t.rank < 2)
                Detail::throw_operand_error(op, operand, "has rank {}, a matrix needs rank 2 or more",
                                            t.rank);

            require_storage(t, op, operand);
            require_device_dtype(t, op, operand);
            require_contiguous(t, op, operand);

            MatrixOperand m;
            m.rank = t.rank;
            m.rows = t.shape[t.rank - 2];
            m.cols = t.shape[t.rank - 1];
            m.layout = t.layout;
            m.matrix_elements = m.rows * m.cols;
            m.leading_dimension = (t.layout == Core::Layout::RowMajor) ? m.cols : m.rows;

            for (int32_t i = 0; i < t.rank - 2; ++i)
                m.batch_count *= t.shape[i];

            return m;
        }

        /**
         * @brief Returns the element distance between consecutive matrices of operand within a call.
         * @param batch_count The number of matrices the call computes.
         * @return 0 when operand has rank 2 or a batch count of 1, matrix_elements otherwise.
         * @throws std::invalid_argument when operand is batched with a different batch count.
         */
        [[nodiscard]] inline int64_t batch_stride(const MatrixOperand& m, int64_t batch_count,
                                                  const char* op, const char* operand)
        {
            if (m.rank == 2 || m.batch_count == 1)
                return 0;

            if (m.batch_count == batch_count)
                return m.matrix_elements;

            Detail::throw_operand_error(op, operand,
                                        "holds {} matrices, but the call computes {}; a batched "
                                        "operand must match that count, hold one matrix or have "
                                        "rank 2",
                                        m.batch_count, batch_count);
        }

        /**
         * @brief Throws unless two operands that both hold more than one matrix share their batch axes.
         * @pre a and b have passed describe_matrix_operand, which gave ma and mb.
         * @throws std::invalid_argument when the axes before the last two differ in count or size.
         */
        inline void require_same_batch_axes(const Tensor& a, const MatrixOperand& ma,
                                            const Tensor& b, const MatrixOperand& mb,
                                            const char* op, const char* operand_a,
                                            const char* operand_b)
        {
            if (ma.batch_count <= 1 || mb.batch_count <= 1)
                return;

            bool same = a.rank == b.rank;
            for (int32_t i = 0; same && i < a.rank - 2; ++i)
                same = a.shape[i] == b.shape[i];

            if (!same)
                Detail::throw_operand_error(op, operand_a,
                                            "has batch axes that differ from those of '{}'; two "
                                            "batched operands must share every axis before the "
                                            "last two",
                                            operand_b);
        }

        /**
         * @brief Throws unless two matrix operands share one layout.
         * @throws std::invalid_argument naming both layouts on a mismatch.
         */
        inline void require_same_layout(const MatrixOperand& a, const MatrixOperand& b,
                                        const char* op, const char* operand_a, const char* operand_b)
        {
            if (a.layout != b.layout)
                Detail::throw_operand_error(op, operand_a,
                                            "is {}, but '{}' is {}; every operand of one call must "
                                            "share a layout",
                                            a.layout == Core::Layout::RowMajor ? "row-major"
                                                                                : "column-major",
                                            operand_b,
                                            b.layout == Core::Layout::RowMajor ? "row-major"
                                                                                : "column-major");
        }

        /**
         * @brief Throws unless one matrix extent of an operand equals the extent the call needs.
         * @param what Names the extent in the message, such as "rows of op(A)".
         * @throws std::invalid_argument naming both extents on a mismatch.
         */
        inline void require_matrix_extent(int64_t actual, int64_t expected, const char* what,
                                          const char* op, const char* operand)
        {
            if (actual != expected)
                Detail::throw_operand_error(op, operand, "has {} {}, the call needs {}", actual, what,
                                            expected);
        }

        /**
         * @brief Throws unless a matrix operand holds exactly one matrix of rank 2.
         * @throws std::invalid_argument naming the rank.
         */
        inline void require_unbatched(const MatrixOperand& m, const char* op, const char* operand)
        {
            if (m.rank != 2)
                Detail::throw_operand_error(op, operand, "has rank {}; a Level 2 matrix has rank 2",
                                            m.rank);
        }

        /**
         * @brief Throws unless a matrix operand has as many rows as columns.
         * @throws std::invalid_argument naming both extents.
         */
        inline void require_square(const MatrixOperand& m, const char* op, const char* operand)
        {
            if (m.rows != m.cols)
                Detail::throw_operand_error(op, operand,
                                            "is {} x {}; the operation needs a square matrix",
                                            m.rows, m.cols);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
