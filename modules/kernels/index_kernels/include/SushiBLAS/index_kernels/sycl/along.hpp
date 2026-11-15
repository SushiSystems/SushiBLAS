/**************************************************************************/
/* along.hpp                                                              */
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
 * @file along.hpp
 * @brief Defines the SYCL kernels that pick one element per row and write it back.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/index_kernels/spans.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Copies `picked[r] = values[r, index[r]]`, one work-item per row.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event take_along(sycl::queue& queue, const MatrixSpan& values_span,
                                                 const T* values, const Index* index,
                                                 int64_t index_stride, T* picked,
                                                 int64_t picked_stride,
                                                 const std::vector<sycl::event>& deps)
            {
                if (values_span.rows <= 0)
                    return sycl::event();

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(values_span.rows),
                                    [=](std::size_t row)
                    {
                        const int64_t r = static_cast<int64_t>(row);
                        const Index selected = index[r * index_stride];

                        picked[r * picked_stride] =
                            is_in_range(selected, values_span.cols)
                                ? values[values_span.offset(r, selected)] : T(0);
                    });
                });
            }

            /**
             * @brief Writes `(c == index[r]) ? picked[r] : 0`, one work-item per element.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            [[nodiscard]] sycl::event put_along(sycl::queue& queue, const T* picked,
                                                int64_t picked_stride, const Index* index,
                                                int64_t index_stride,
                                                const MatrixSpan& values_span, T* values,
                                                const std::vector<sycl::event>& deps)
            {
                const int64_t total = values_span.rows * values_span.cols;
                if (total <= 0)
                    return sycl::event();

                const int64_t cols = values_span.cols;

                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    Launch::indexed(h, queue, static_cast<std::size_t>(total),
                                    [=](std::size_t flat)
                    {
                        const int64_t r = static_cast<int64_t>(flat) / cols;
                        const int64_t c = static_cast<int64_t>(flat) % cols;
                        const int64_t selected = static_cast<int64_t>(index[r * index_stride]);

                        values[values_span.offset(r, c)] =
                            (c == selected) ? picked[r * picked_stride] : T(0);
                    });
                });
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
