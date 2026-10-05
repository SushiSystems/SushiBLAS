/****************************************************************/
/* upsample.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file upsample.hpp
 * @brief Defines the reference kernels of nearest-neighbour upsampling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Repeats each input pixel into a `scale_h` by `scale_w` block.
             * @see kernels/README.md
             */
            template<typename T>
            void upsample_nearest(const Spatial::WindowSpan& span,
                                  int64_t scale_h, int64_t scale_w,
                                  const T* in, T* out)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        const int64_t h_in = h_o / scale_h;

                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            const int64_t w_in = w_o / scale_w;

                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                out[n * span.out_batch_stride + h_o * span.out_row_stride
                                    + w_o * span.out_col_stride + c * span.out_channel_stride] =
                                    in[n * span.in_batch_stride + h_in * span.in_row_stride
                                       + w_in * span.in_col_stride + c * span.in_channel_stride];
                            }
                        }
                    }
                }
            }

            /**
             * @brief Sums each output block back onto the pixel that produced it.
             * @param dx The input gradient; every pixel is written, so it needs no prior clear.
             * @see kernels/README.md
             */
            template<typename T>
            void upsample_nearest_backward(const Spatial::WindowSpan& span,
                                           int64_t scale_h, int64_t scale_w,
                                           const T* dy, T* dx)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                T total = T(0);

                                for (int64_t dh = 0; dh < scale_h; ++dh)
                                {
                                    for (int64_t dw = 0; dw < scale_w; ++dw)
                                    {
                                        total += dy[n * span.out_batch_stride
                                                  + (h * scale_h + dh) * span.out_row_stride
                                                  + (w * scale_w + dw) * span.out_col_stride
                                                  + c * span.out_channel_stride];
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
