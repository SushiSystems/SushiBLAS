/**************************************************************************/
/* gather.hpp                                                             */
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
 * @file gather.hpp
 * @brief Declares RowIndexOps, the gathers and their transposes that select by an integer index.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    /**
     * @class RowIndexOps
     * @brief Records gathers and their transposes by an integer index.
     * @note An index outside its range selects nothing: a gather reads it as zero and a
     *       transpose skips it, on both paths and in every build type.
     * @see modules/operations/gather/README.md
     */
    class RowIndexOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit RowIndexOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            RowIndexOps(const RowIndexOps&) = default;
            RowIndexOps& operator=(const RowIndexOps&) = delete;
            RowIndexOps(RowIndexOps&&) = default;
            RowIndexOps& operator=(RowIndexOps&&) = delete;
            ~RowIndexOps() = default;

            /**
             * @brief Copies table rows: `rows[r, d] = table[index[r], d]`.
             * @param table A `[V, D]` matrix of any element type but UINT32.
             * @param index A rank-1 INT32 or INT64 tensor of `R` positions in `[0, V)`.
             * @param rows The `[R, D]` result, of table's dtype; every element is written.
             * @throws std::invalid_argument at record time on a rank, dtype, extent or overlap.
             * @see modules/operations/gather/README.md
             */
            TaskHandle take_rows(const Tensor& table, const Tensor& index, Tensor& rows);

            /**
             * @brief Sums rows into a table: `beta * table` plus the rows each index selects.
             * @param rows A `[R, D]` matrix of HALF, FLOAT32 or FLOAT64.
             * @param index A rank-1 INT32 or INT64 tensor of `R` positions in `[0, V)`.
             * @param table The `[V, D]` result, of rows' dtype; every element is written.
             * @param beta Exactly zero does not read table.
             * @see modules/operations/gather/README.md
             */
            TaskHandle add_rows(const Tensor& rows, const Tensor& index, Tensor& table,
                                double beta = 0.0);

            /**
             * @brief Picks one element per row: `picked[r] = values[r, index[r]]`.
             * @param values A `[R, C]` matrix of any element type but UINT32.
             * @param index A rank-1 INT32 or INT64 tensor of `R` positions in `[0, C)`.
             * @param picked The rank-1 result of `R` elements, of values' dtype.
             * @throws std::invalid_argument at record time on a rank, dtype, extent or overlap.
             * @see modules/operations/gather/README.md
             */
            TaskHandle take_along(const Tensor& values, const Tensor& index, Tensor& picked);

            /**
             * @brief Writes one element per row, zero elsewhere: `(c == index[r]) ? picked[r] : 0`.
             * @param picked A rank-1 tensor of `R` elements.
             * @param index A rank-1 INT32 or INT64 tensor of `R` positions in `[0, C)`.
             * @param values The `[R, C]` result, of picked's dtype; every element is written.
             * @throws std::invalid_argument at record time on a faulty operand.
             * @see modules/operations/gather/README.md
             */
            TaskHandle put_along(const Tensor& picked, const Tensor& index, Tensor& values);

        private:
            /// @brief The recorder every task of this view is recorded on.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
