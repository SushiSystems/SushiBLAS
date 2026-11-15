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
 * @brief Defines the reference kernels that pick one element per row and write it back.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/index_kernels/spans.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Copies `picked[r] = values[r, index[r]]`, or zero for an index outside it.
             * @param index_stride Elements between two indices.
             * @param picked_stride Elements between two picked elements.
             * @pre picked overlaps neither values nor index.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            void take_along(const MatrixSpan& values_span, const T* values, const Index* index,
                            int64_t index_stride, T* picked, int64_t picked_stride)
            {
                for (int64_t r = 0; r < values_span.rows; ++r)
                {
                    const Index selected = index[r * index_stride];
                    picked[r * picked_stride] =
                        is_in_range(selected, values_span.cols)
                            ? values[values_span.offset(r, selected)] : T(0);
                }
            }

            /**
             * @brief Writes `values[r, c] = (c == index[r]) ? picked[r] : 0`.
             * @pre values overlaps neither picked nor index.
             * @see modules/kernels/index_kernels/README.md
             */
            template<typename T, typename Index>
            void put_along(const T* picked, int64_t picked_stride, const Index* index,
                           int64_t index_stride, const MatrixSpan& values_span, T* values)
            {
                for (int64_t r = 0; r < values_span.rows; ++r)
                {
                    const int64_t selected = static_cast<int64_t>(index[r * index_stride]);

                    for (int64_t c = 0; c < values_span.cols; ++c)
                        values[values_span.offset(r, c)] =
                            (c == selected) ? picked[r * picked_stride] : T(0);
                }
            }
        } // namespace Reference
    } // namespace Kernels
} // namespace SushiBLAS
