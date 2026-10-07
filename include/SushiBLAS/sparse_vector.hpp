/**************************************************************************/
/* sparse_vector.hpp                                                      */
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
 * @file sparse_vector.hpp
 * @brief Declares SparseVector, a vector held as the positions and values of its stored entries.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/sparse_types.hpp>

namespace SushiBLAS
{
    /**
     * @brief Holds a sparse vector in compressed form: two tensors, a length and an index base.
     * @see include/SushiBLAS/README.md
     */
    struct SparseVector
    {
        /** @brief The length of the dense vector this one compresses. */
        int64_t size = 0;

        /** @brief The stored value that names position 0. */
        Core::IndexBase index_base = Core::IndexBase::Zero;

        /** @brief One position per stored entry: rank 1, INT32 or INT64, no position twice. */
        Tensor indices;

        /** @brief One value per stored entry, in the order of indices: rank 1. */
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

        /** @brief Returns the element type of the stored positions. */
        [[nodiscard]] Core::DataType index_dtype() const noexcept
        {
            return indices.dtype;
        }
    };
} // namespace SushiBLAS
