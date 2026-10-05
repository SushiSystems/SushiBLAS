/****************************************************************/
/* pool_max.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file pool_max.hpp
 * @brief Defines the reference kernels of max pooling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <limits>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Takes each window's maximum, and records where it came from.
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
                                    const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                    if (h_in < 0 || h_in >= span.in_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                        if (w_in < 0 || w_in >= span.in_w) continue;

                                        const T candidate = in[n * span.in_batch_stride
                                                             + h_in * span.in_row_stride
                                                             + w_in * span.in_col_stride
                                                             + c * span.in_channel_stride];

                                        if (candidate > best)
                                        {
                                            best = candidate;
                                            best_index = ((n * span.in_h + h_in) * span.in_w + w_in)
                                                       * span.channels + c;
                                        }
                                    }
                                }

                                const int64_t at = n * span.out_batch_stride
                                                 + h_o * span.out_row_stride
                                                 + w_o * span.out_col_stride
                                                 + c * span.out_channel_stride;
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

                                T total = T(0);

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_num = h + g.pad_top - r * g.dilation_h;
                                    if (h_num < 0 || h_num % g.stride_h != 0) continue;
                                    const int64_t h_o = h_num / g.stride_h;
                                    if (h_o >= span.out_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_num = w + g.pad_left - s * g.dilation_w;
                                        if (w_num < 0 || w_num % g.stride_w != 0) continue;
                                        const int64_t w_o = w_num / g.stride_w;
                                        if (w_o >= span.out_w) continue;

                                        const int64_t at = n * span.out_batch_stride
                                                         + h_o * span.out_row_stride
                                                         + w_o * span.out_col_stride
                                                         + c * span.out_channel_stride;

                                        if (indices[at] == mine) total += dy[at];
                                    }
                                }

                                dx[n * span.in_batch_stride + h * span.in_row_stride
                                   + w * span.in_col_stride + c * span.in_channel_stride] = total;
                            }
                        }
                    }
                }
            }
        }
    }
}
