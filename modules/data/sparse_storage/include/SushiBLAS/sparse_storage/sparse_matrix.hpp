/**************************************************************************/
/* sparse_matrix.hpp                                                      */
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
 * @file sparse_matrix.hpp
 * @brief Declares SparseMatrix, a matrix held in compressed sparse row form.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds a CSR matrix: three tensors, a shape and an index base.
     * @see modules/README.md
     */
    struct SparseMatrix
    {
        /** @brief The number of rows. */
        int64_t rows = 0;

        /** @brief The number of columns. */
        int64_t cols = 0;

        /** @brief The stored value that names row 0 and column 0. */
        Core::IndexBase index_base = Core::IndexBase::Zero;

        /** @brief Where each row starts among the entries: rank 1, rows + 1 INT32 or INT64. */
        Tensor row_offsets;

        /** @brief The column of each stored entry: rank 1, the dtype of row_offsets. */
        Tensor column_indices;

        /** @brief The value of each stored entry, in the order of column_indices: rank 1. */
        Tensor values;

        /** @brief Returns the number of stored entries. */
        [[nodiscard]] int64_t nnz() const noexcept
        {
            return values.num_elements;
        }

        /** @brief Returns the element type of the stored values. */
        [[nodiscard]] Core::DataType dtype() const noexcept
        {
            return values.dtype;
        }

        /** @brief Returns the element type of the row offsets and the column indices. */
        [[nodiscard]] Core::DataType index_dtype() const noexcept
        {
            return row_offsets.dtype;
        }
    };
} // namespace SushiBLAS
