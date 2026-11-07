/**************************************************************************/
/* pool_max.hpp                                                           */
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
 * @file pool_max.hpp
 * @brief Defines the reference kernels of max pooling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <limits>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/window_index.hpp>
#include <SushiBLAS/spatial_kernels/pool_max_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Takes each window's maximum, or its first NaN, and records where it came from.
             * @param indices The argmax buffer's first element; laid out exactly like @p out.
             * @see kernels/README.md
             */
            template<typename T>
            void pool_max(const Spatial::WindowSpan& span, const T* in, T* out, int64_t* indices)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                T best = std::numeric_limits<T>::lowest();
                                int64_t best_index = -1;

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_in = Indexing::gather_coord(h_o, g.stride_h, g.pad_top, r, g.dilation_h);
                                    if (!Indexing::in_extent(h_in, span.in_h))
                                        continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_in = Indexing::gather_coord(w_o, g.stride_w, g.pad_left, s, g.dilation_w);
                                        if (!Indexing::in_extent(w_in, span.in_w))
                                            continue;

                                        const T candidate = in[Indexing::input_strides(span).offset(n, h_in, w_in, c)];

                                        // The first live tap is taken; a NaN tap wins and stays.
                                        if (Indexing::takes_candidate(best_index, best, candidate))
                                        {
                                            best = candidate;
                                            best_index = ((n * span.in_h + h_in) * span.in_w + w_in)
                                                       * span.channels + c;
                                        }
                                    }
                                }

                                const int64_t at = Indexing::output_strides(span).offset(n, h_o, w_o, c);
                                out[at] = best;
                                indices[at] = best_index;
                            }
                        }
                    }
                }
            }

            /**
             * @brief Routes each output gradient back to the element it was taken from.
             * @param indices The argmax buffer `pool_max` wrote.
             * @param dx The input gradient; every pixel is written, so it needs no prior clear.
             * @see kernels/README.md
             */
            template<typename T>
            void pool_max_backward(const Spatial::WindowSpan& span, const T* dy,
                                   const int64_t* indices, T* dx)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                const int64_t mine =
                                    ((n * span.in_h + h) * span.in_w + w) * span.channels + c;

                                Accumulator<T> total = Accumulator<T>(0);

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_o = Indexing::scatter_coord(h, g.stride_h, g.pad_top, r, g.dilation_h, span.out_h);
                                    if (h_o < 0)
                                        continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_o = Indexing::scatter_coord(w, g.stride_w, g.pad_left, s, g.dilation_w, span.out_w);
                                        if (w_o < 0)
                                            continue;

                                        const int64_t at = Indexing::output_strides(span).offset(n, h_o, w_o, c);

                                        if (indices[at] == mine)
                                            total += Accumulator<T>(dy[at]);
                                    }
                                }

                                dx[Indexing::input_strides(span).offset(n, h, w, c)] =
                                    static_cast<T>(total);
                            }
                        }
                    }
                }
            }
        }
    }
}
