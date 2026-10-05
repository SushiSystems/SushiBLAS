/****************************************************************/
/* pool_avg.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
             * @brief Averages each window, dividing by the window's **area**.
             *
             * The divisor is `kernel_h * kernel_w`, counting padded taps that
             * contributed nothing - PyTorch's `count_include_pad=True`, which is
             * its default and therefore what a network ported from it expects.
             * The alternative, dividing by the number of in-bounds taps, is
             * defensible and produces different numbers at every border pixel of
             * every padded pool; the choice is invisible in any test that does
             * not pad, which is why it is stated here and pinned by a test named
             * for it.
             *
             * Summed in ascending `(r, s)` order, so the additions are a pure
             * function of the geometry and two runs perform them identically.
             *
             * @tparam T The element type.
             * @param span The window layout, in element counts.
             * @param in The input image's first element.
             * @param out The output image's first element.
             */
            template<typename T>
            void pool_avg(const Spatial::WindowSpan& span, const T* in, T* out)
            {
                const Spatial::ConvGeometry& g = span.geometry;
                const T area = static_cast<T>(g.kernel_h * g.kernel_w);

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                T total = T(0);

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                    if (h_in < 0 || h_in >= span.in_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                        if (w_in < 0 || w_in >= span.in_w) continue;

                                        total += in[n * span.in_batch_stride
                                                  + h_in * span.in_row_stride
                                                  + w_in * span.in_col_stride
                                                  + c * span.in_channel_stride];
                                    }
                                }

                                out[n * span.out_batch_stride + h_o * span.out_row_stride
                                    + w_o * span.out_col_stride + c * span.out_channel_stride] =
                                    total / area;
                            }
                        }
                    }
                }
            }

            /**
             * @brief Spreads each output gradient evenly over the window it came from.
             *
             * The same divisibility gather as @ref pool_max_backward, and
             * deliberately **not** the same function behind a mode flag: this
             * one adds `dy / area` for every window covering the pixel, with no
             * index to test. The two share a loop shape and nothing else - one
             * selects through a recorded index, the other broadcasts a constant -
             * and folding them together would put a branch in the inner loop to
             * express a similarity that is only skin deep.
             *
             * The divisor is the window area, matching @ref pool_avg.
             *
             * @tparam T The element type.
             * @param span The window layout, in element counts.
             * @param dy The output gradient's first element.
             * @param dx The input gradient's first element; every pixel is
             *        written, so it needs no prior clear.
             */
            template<typename T>
            void pool_avg_backward(const Spatial::WindowSpan& span, const T* dy, T* dx)
            {
                const Spatial::ConvGeometry& g = span.geometry;
                const T area = static_cast<T>(g.kernel_h * g.kernel_w);

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
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

                                        total += dy[n * span.out_batch_stride
                                                  + h_o * span.out_row_stride
                                                  + w_o * span.out_col_stride
                                                  + c * span.out_channel_stride] / area;
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
