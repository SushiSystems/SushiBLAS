/**************************************************************************/
/* sparse_vector_operand.hpp                                              */
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
 * @file sparse_vector_operand.hpp
 * @brief Declares SparseVectorOperand, the geometry of one sparse vector operand of a call.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/sparse_vector.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/sparse/index_check.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds the length, the entry count and the first index of one sparse vector.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        struct SparseVectorOperand
        {
            int64_t size = 0;        /**< Length of the dense vector the operand compresses. */
            int64_t nnz = 0;         /**< Number of stored entries. */
            int64_t first_index = 0; /**< The stored value that names position 0. */
        };

        /**
         * @brief Returns the geometry of a sparse vector operand.
         * @throws std::invalid_argument on an index base outside its type, a negative size, an
         *         array that is not a contiguous rank-1 tensor with storage, an index dtype other
         *         than INT32 or INT64 or too narrow for the size, arrays of different lengths or
         *         more entries than the size; UnsupportedExecution on a dtype the device lacks.
         * @see include/SushiBLAS/engine/sparse/README.md
         */
        [[nodiscard]] inline SparseVectorOperand describe_sparse_vector(const SparseVector& x,
                                                                        const char* op,
                                                                        const char* operand)
        {
            require_index_base(x.index_base, op, operand);
            if (x.size < 0)
                Detail::throw_operand_error(op, operand, "has the size {}; a size is not negative",
                                            x.size);

            SparseVectorOperand v;
            v.size = x.size;
            v.nnz = x.values.num_elements;
            v.first_index = Core::first_index(x.index_base);

            require_index_array(x.indices, v.nnz, v.size - 1 + v.first_index, "indices", op,
                                operand);
            require_compressed_array(x.values, v.nnz, "values", op, operand);
            require_device_dtype(x.values, op, operand);

            if (v.nnz > v.size)
                Detail::throw_operand_error(op, operand,
                                            "stores {} entries in a vector of size {}; a position "
                                            "is stored at most once",
                                            v.nnz, v.size);
            return v;
        }
    } // namespace Dispatch
} // namespace SushiBLAS
