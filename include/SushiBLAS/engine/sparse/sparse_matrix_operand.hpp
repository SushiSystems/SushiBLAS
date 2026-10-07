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
 * @file sparse_matrix_operand.hpp
 * @brief Declares SparseMatrixOperand, the geometry of one CSR matrix operand of a call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/sparse_matrix.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/sparse/index_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the shape, the entry count and the first index of one CSR matrix.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        struct SparseMatrixOperand
        {
            int64_t rows = 0;        /**< Number of rows. */
            int64_t cols = 0;        /**< Number of columns. */
            int64_t nnz = 0;         /**< Number of stored entries. */
            int64_t first_index = 0; /**< The stored value that names row 0 and column 0. */
        };

        /**
         * @brief Returns the geometry of a CSR matrix operand.
         * @throws std::invalid_argument on an index base outside its type, a negative extent,
         *         entries without rows or columns, an array that is not a contiguous rank-1 tensor
         *         of the length the shape needs, an index dtype that is not INT32 or INT64, differs
         *         between the index arrays or is too narrow, or a value dtype the device lacks.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        [[nodiscard]] inline SparseMatrixOperand describe_sparse_matrix(const SparseMatrix& A,
                                                                        const char* op,
                                                                        const char* operand)
        {
            require_index_base(A.index_base, op, operand);
            if (A.rows < 0 || A.cols < 0)
                Detail::throw_operand_error(op, operand, "is {} x {}; an extent is not negative",
                                            A.rows, A.cols);

            SparseMatrixOperand m;
            m.rows = A.rows;
            m.cols = A.cols;
            m.nnz = A.values.num_elements;
            m.first_index = Core::first_index(A.index_base);

            if (m.nnz > 0 && (m.rows == 0 || m.cols == 0))
                Detail::throw_operand_error(op, operand, "stores {} entries in a {} x {} matrix",
                                            m.nnz, m.rows, m.cols);
            if (A.row_offsets.num_elements - 1 != m.rows)
                Detail::throw_operand_error(op, operand,
                                            "has {} row offsets for {} rows; a CSR matrix has one "
                                            "offset more than it has rows",
                                            A.row_offsets.num_elements, m.rows);

            require_index_array(A.row_offsets, A.row_offsets.num_elements, m.nnz + m.first_index,
                                "row offsets", op, operand);
            require_index_array(A.column_indices, m.nnz, m.cols - 1 + m.first_index,
                                "column indices", op, operand);
            if (A.column_indices.dtype != A.row_offsets.dtype)
                Detail::throw_operand_error(op, operand,
                                            "has row offsets of dtype {} and column indices of "
                                            "dtype {}; one matrix has one index dtype",
                                            Core::to_string(A.row_offsets.dtype),
                                            Core::to_string(A.column_indices.dtype));

            require_compressed_array(A.values, m.nnz, "values", op, operand);
            require_device_dtype(A.values, op, operand);
            return m;
        }
    } // namespace Dispatch
} // namespace SushiBLAS
