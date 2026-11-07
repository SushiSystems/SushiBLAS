/**************************************************************************/
/* sparse_matrix_operand.hpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file sparse_matrix_operand.hpp
 * @brief Declares SparseMatrixOperand, the geometry of one CSR matrix operand of a call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/sparse/index_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the shape, the entry count and the first index of one CSR matrix.
         * @see modules/operations/sparse/README.md
         */
        struct SparseMatrixOperand
        {
            int64_t rows = 0;        /**< Number of rows. */
            int64_t cols = 0;        /**< Number of columns. */
            int64_t nnz = 0;         /**< Number of stored entries. */
            int64_t first_index = 0; /**< The stored value that names row 0 and column 0. */
        };

        /**
         * @brief Pairs the extents and the index base of a CSR matrix with its three arrays and
         *        the facts of their storage, so a test describes a matrix without a runtime.
         * @note The arrays are held by reference through PlacedTensor and must outlive the pair.
         * @see modules/operations/sparse/README.md
         */
        struct PlacedSparseMatrix
        {
            int64_t rows;               /**< Number of rows. */
            int64_t cols;               /**< Number of columns. */
            Core::IndexBase index_base; /**< The base the stored indices count from. */
            PlacedTensor row_offsets;   /**< The row offsets. */
            PlacedTensor column_indices; /**< The column of each stored entry. */
            PlacedTensor values;        /**< The stored values. */
        };

        /**
         * @brief Returns the geometry of a CSR matrix operand.
         * @throws std::invalid_argument on an index base outside its type, a negative extent,
         *         entries without rows or columns, an array that is not a contiguous rank-1 tensor
         *         of the needed length, an index dtype that is not INT32 or INT64, differs between
         *         the arrays or is too narrow; UnsupportedExecution on a dtype the device lacks.
         * @see modules/operations/sparse/README.md
         */
        [[nodiscard]] inline SparseMatrixOperand describe_sparse_matrix(
            const PlacedSparseMatrix& A, const char* op, const char* operand)
        {
            require_index_base(A.index_base, op, operand);
            if (A.rows < 0 || A.cols < 0)
                Detail::throw_operand_error(op, operand, "is {} x {}; an extent is not negative",
                                            A.rows, A.cols);

            SparseMatrixOperand m;
            m.rows = A.rows;
            m.cols = A.cols;
            m.nnz = A.values.tensor.num_elements;
            m.first_index = Core::first_index(A.index_base);

            const int64_t offset_count = A.row_offsets.tensor.num_elements;
            if (m.nnz > 0 && (m.rows == 0 || m.cols == 0))
                Detail::throw_operand_error(op, operand, "stores {} entries in a {} x {} matrix",
                                            m.nnz, m.rows, m.cols);
            if (offset_count - 1 != m.rows)
                Detail::throw_operand_error(op, operand,
                                            "has {} row offsets for {} rows; a CSR matrix has one "
                                            "offset more than it has rows",
                                            offset_count, m.rows);

            require_index_array(A.row_offsets, offset_count, m.nnz + m.first_index, "row offsets",
                                op, operand);
            require_index_array(A.column_indices, m.nnz, m.cols - 1 + m.first_index,
                                "column indices", op, operand);
            if (A.column_indices.tensor.dtype != A.row_offsets.tensor.dtype)
                Detail::throw_operand_error(op, operand,
                                            "has row offsets of dtype {} and column indices of "
                                            "dtype {}; one matrix has one index dtype",
                                            Core::to_string(A.row_offsets.tensor.dtype),
                                            Core::to_string(A.column_indices.tensor.dtype));

            require_compressed_array(A.values, m.nnz, "values", op, operand);
            require_device_dtype(A.values.tensor, A.values.storage, op, operand);
            return m;
        }

        /**
         * @brief Returns the geometry of a CSR matrix operand.
         * @throws std::invalid_argument and UnsupportedExecution as the overload that takes the
         *         facts of the storage does.
         * @see modules/operations/sparse/README.md
         */
        [[nodiscard]] inline SparseMatrixOperand describe_sparse_matrix(const SparseMatrix& A,
                                                                        const char* op,
                                                                        const char* operand)
        {
            return describe_sparse_matrix(
                PlacedSparseMatrix{A.rows, A.cols, A.index_base, placed(A.row_offsets),
                                   placed(A.column_indices), placed(A.values)},
                op, operand);
        }
    } // namespace Dispatch
} // namespace SushiBLAS
