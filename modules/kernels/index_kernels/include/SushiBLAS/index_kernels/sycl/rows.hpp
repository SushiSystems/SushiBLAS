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
 * @brief Defines the SYCL kernels that copy table rows by index and sum rows into a table.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/index_kernels/spans.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Copies `rows[r, d] = table[index[r], d]`, one work-item per element of rows.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event take_rows(sycl::queue& queue, const MatrixSpan& table_span,
                                                const T* table, const Index* index,
                                                int64_t index_stride,
                                                const MatrixSpan& rows_span, T* rows,
                                                const std::vector<sycl::event>& deps)
            {
                const int64_t total = rows_span.rows * rows_span.cols;
                if (total <= 0)
                    return sycl::event();

                const int64_t cols = rows_span.cols;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(total),
                                    [=](std::size_t flat)
                    {
                        const int64_t r = static_cast<int64_t>(flat) / cols;
                        const int64_t d = static_cast<int64_t>(flat) % cols;
                        const Index selected = index[r * index_stride];

                        rows[rows_span.offset(r, d)] =
                            is_in_range(selected, table_span.rows)
                                ? table[table_span.offset(selected, d)] : T(0);
                    });
                });
            }

            /**
             * @brief Computes `table = beta * table` plus every `rows[r, d]` with `index[r] == v`.
             * @param beta Exactly zero does not read table.
             * @param work One Accumulator<T> per element of table's span when
             *        RunningSolution<T>::K_NEEDS_WORKSPACE holds; never read otherwise.
             * @pre table overlaps neither rows, index nor work.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event add_rows(sycl::queue& queue, const MatrixSpan& rows_span,
                                               const T* rows, const Index* index,
                                               int64_t index_stride, const MatrixSpan& table_span,
                                               T* table, double beta, Accumulator<T>* work,
                                               const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                if (table_span.rows <= 0 || table_span.cols <= 0)
                    return sycl::event();

                const bool keeps_table = !(beta == 0.0);
                const W scale = static_cast<W>(beta);

                // One work-item owns one column of table, so no element has two writers.
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(table_span.cols),
                                    [=](std::size_t column)
                    {
                        const int64_t d = static_cast<int64_t>(column);
                        const RunningSolution<T> sums(table, work, 0);

                        for (int64_t v = 0; v < table_span.rows; ++v)
                        {
                            const int64_t at = table_span.offset(v, d);
                            sums.set(at, keeps_table ? scale * W(table[at]) : W(0));
                        }

                        // The rows are added in ascending r, as the reference adds them.
                        for (int64_t r = 0; r < rows_span.rows; ++r)
                        {
                            const Index selected = index[r * index_stride];
                            if (!is_in_range(selected, table_span.rows))
                                continue;

                            const int64_t at = table_span.offset(selected, d);
                            sums.set(at, sums.get(at) + W(rows[rows_span.offset(r, d)]));
                        }
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
