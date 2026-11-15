/**************************************************************************/
/* rows.hpp                                                               */
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
 * @file rows.hpp
 * @brief Defines the reference kernels that copy table rows by index and sum rows into a table.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/index_kernels/spans.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Copies `rows[r, d] = table[index[r], d]`, or zeros for an index outside it.
             * @param index_stride Elements between two indices.
             * @pre rows overlaps neither table nor index.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            void take_rows(const MatrixSpan& table_span, const T* table, const Index* index,
                           int64_t index_stride, const MatrixSpan& rows_span, T* rows)
            {
                for (int64_t r = 0; r < rows_span.rows; ++r)
                {
                    const Index selected = index[r * index_stride];
                    const bool is_present = is_in_range(selected, table_span.rows);

                    for (int64_t d = 0; d < rows_span.cols; ++d)
                        rows[rows_span.offset(r, d)] =
                            is_present ? table[table_span.offset(selected, d)] : T(0);
                }
            }

            /**
             * @brief Computes `table = beta * table` plus every `rows[r, d]` with `index[r] == v`.
             * @param beta Exactly zero does not read table.
             * @pre table overlaps neither rows nor index.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            void add_rows(const MatrixSpan& rows_span, const T* rows, const Index* index,
                          int64_t index_stride, const MatrixSpan& table_span, T* table,
                          double beta)
            {
                using RW = ReferenceAccumulator<T>;

                const bool keeps_table = !(beta == 0.0);
                std::vector<ReferenceSum<T>> sums;

                for (int64_t d = 0; d < table_span.cols; ++d)
                {
                    sums.assign(static_cast<std::size_t>(table_span.rows), ReferenceSum<T>());

                    // The rows are added in ascending r; see kernels/README.md.
                    for (int64_t r = 0; r < rows_span.rows; ++r)
                    {
                        const Index selected = index[r * index_stride];
                        if (is_in_range(selected, table_span.rows))
                            sums[static_cast<std::size_t>(selected)].add(
                                RW(rows[rows_span.offset(r, d)]));
                    }

                    for (int64_t v = 0; v < table_span.rows; ++v)
                    {
                        T& out = table[table_span.offset(v, d)];
                        const RW sum = sums[static_cast<std::size_t>(v)].value();
                        out = keeps_table ? static_cast<T>(RW(beta) * RW(out) + sum)
                                          : static_cast<T>(sum);
                    }
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
