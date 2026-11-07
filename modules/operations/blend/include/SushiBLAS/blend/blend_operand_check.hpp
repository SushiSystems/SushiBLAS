/**************************************************************************/
/* blend_operand_check.hpp                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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

/**
 * @file blend_operand_check.hpp
 * @brief Declares the record-time operand checks of BlendOps::blend_transforms, which read
 *        shapes and storage facts and so run without a runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/operand_check.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Throws unless a tensor is dense and row-major, so its rows are contiguous.
         * @throws std::invalid_argument on a view, a slice or a column-major tensor.
         * @see modules/operations/blend/README.md
         */
        inline void require_dense_row_major(const Tensor& t, const char* name, const char* operand)
        {
            Dispatch::require_contiguous(t, name, operand);
            Dispatch::require_operand(t.layout == Core::Layout::RowMajor, name, operand,
                                      "is column-major; it must be dense and row-major");
        }

        /**
         * @brief Throws unless the five operands of a blend are a well-formed call.
         * @param name The operation id the messages name.
         * @throws std::invalid_argument naming the operand and the rule it breaks;
         *         UnsupportedExecution for a dtype the device of an operand cannot compute.
         * @see modules/operations/blend/README.md
         */
        inline void check_blend_operands(const Dispatch::PlacedTensor& points,
                                         const Dispatch::PlacedTensor& transforms,
                                         const Dispatch::PlacedTensor& indices,
                                         const Dispatch::PlacedTensor& weights,
                                         const Dispatch::PlacedTensor& out, const char* name)
        {
            Dispatch::require_dtype(transforms.tensor, points.tensor.dtype, name, "transforms");
            Dispatch::require_dtype(weights.tensor, points.tensor.dtype, name, "weights");
            Dispatch::require_dtype(out.tensor, points.tensor.dtype, name, "out");
            Dispatch::require_operand(indices.tensor.dtype == Core::DataType::Int32 ||
                                          indices.tensor.dtype == Core::DataType::Int64 ||
                                          indices.tensor.dtype == Core::DataType::UInt32,
                                      name, "indices",
                                      "has dtype {}; indices are INT32, INT64 or UINT32",
                                      Core::to_string(indices.tensor.dtype));

            Dispatch::require_rank(points.tensor, 2, name, "points");
            Dispatch::require_rank(transforms.tensor, 3, name, "transforms");
            Dispatch::require_rank(indices.tensor, 2, name, "indices");
            Dispatch::require_rank(weights.tensor, 2, name, "weights");
            Dispatch::require_rank(out.tensor, 2, name, "out");

            const int64_t point_count = points.tensor.shape[0];
            const int64_t components = points.tensor.shape[1];
            const int64_t transform_count = transforms.tensor.shape[0];
            const int64_t transform_rows = transforms.tensor.shape[1];
            const int64_t influences = indices.tensor.shape[1];

            Dispatch::require_operand(components == 3 || components == 4, name, "points",
                                      "has {} values per point; a position has 3 and a "
                                      "homogeneous point 4", components);
            Dispatch::require_operand(point_count >= 1, name, "points", "holds no points");
            Dispatch::require_operand(transform_count >= 1, name, "transforms",
                                      "holds no transforms");
            Dispatch::require_operand(transform_rows == 3 || transform_rows == 4, name,
                                      "transforms",
                                      "has {} rows per transform; a transform has 3 or 4",
                                      transform_rows);
            Dispatch::require_extent(transforms.tensor.shape[2], 4, "columns per transform", name,
                                     "transforms");
            Dispatch::require_operand(influences >= 1, name, "indices",
                                      "holds no influences per point; at least 1 is required");
            Dispatch::require_extent(indices.tensor.shape[0], point_count, "rows, one per point",
                                     name, "indices");
            Dispatch::require_same_shape(weights.tensor, indices.tensor, name, "weights",
                                         "indices");
            Dispatch::require_same_shape(out.tensor, points.tensor, name, "out", "points");

            Dispatch::require_storage(points.tensor, points.storage, name, "points");
            Dispatch::require_storage(transforms.tensor, transforms.storage, name, "transforms");
            Dispatch::require_storage(indices.tensor, indices.storage, name, "indices");
            Dispatch::require_storage(weights.tensor, weights.storage, name, "weights");
            Dispatch::require_storage(out.tensor, out.storage, name, "out");
            require_dense_row_major(points.tensor, name, "points");
            require_dense_row_major(transforms.tensor, name, "transforms");
            require_dense_row_major(indices.tensor, name, "indices");
            require_dense_row_major(weights.tensor, name, "weights");
            require_dense_row_major(out.tensor, name, "out");
            Dispatch::require_device_dtype(points.tensor, points.storage, name, "points");
            Dispatch::require_device_dtype(transforms.tensor, transforms.storage, name,
                                           "transforms");
            Dispatch::require_device_dtype(weights.tensor, weights.storage, name, "weights");
            Dispatch::require_device_dtype(out.tensor, out.storage, name, "out");

            // Each output point is written while its inputs are read, so no byte may be shared.
            Dispatch::require_disjoint(out.storage, points.storage, name, "out", "points");
            Dispatch::require_disjoint(out.storage, transforms.storage, name, "out", "transforms");
            Dispatch::require_disjoint(out.storage, indices.storage, name, "out", "indices");
            Dispatch::require_disjoint(out.storage, weights.storage, name, "out", "weights");
        }
    } // namespace Internal
} // namespace SushiBLAS
