/****************************************************************/
/* col2im.hpp                                                   */
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
             * @brief Scatters an im2col matrix back onto an image by summing overlaps.
             *
             * `im2col`'s transpose, and the gradient of it. Written as a
             * **gather**: the loop nest is over output *pixels*, and each pixel
             * finds the column entries that read it and sums them itself. The
             * scatter form - walk the column matrix, add each entry into the
             * pixel it came from - needs an atomic add on overlapping windows,
             * and a floating-point atomic add has a combination order that
             * varies between runs. SushiBLAS links `sushiruntime_deterministic_fp`
             * PUBLIC, so that is not a trade this library is allowed to make.
             *
             * Which terms are summed and the order they are summed in are pure
             * functions of the geometry, so two runs perform the same additions
             * on the same values in the same order.
             *
             * **The divisibility test is what inverts the stride.** Pixel `h` is
             * read by window row `h_o` through tap `r` exactly when
             * `h_o * stride_h - pad_top + r * dilation_h == h`, so `h_o` exists
             * only if `h + pad_top - r * dilation_h` is non-negative and
             * divisible by `stride_h`. Dropping that test makes every pixel look
             * as though it were read by every tap.
             *
             * Every pixel is written exactly once, including pixels no window
             * reads - those get an explicit zero rather than being left alone,
             * so the destination needs no prior clear.
             *
             * @tparam T The element type.
             * @param span The window layout, in element counts; its `in_*` fields
             *        describe the *output* image here, since the image is what
             *        this operation produces.
             * @param cols The column matrix's addressing.
             * @param in The column matrix's first element.
             * @param out The image's first element.
             */
            template<typename T>
            void col2im(const Spatial::WindowSpan& span, const Spatial::ColumnSpan& cols,
                        const T* in, T* out)
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

                                        const int64_t row = (n * span.out_h + h_o) * span.out_w + w_o;
                                        const int64_t patch = (r * g.kernel_w + s) * span.channels + c;

                                        total += in[row * cols.row_stride + patch * cols.element_stride];
                                    }
                                }

                                out[n * span.in_batch_stride + h * span.in_row_stride
                                    + w * span.in_col_stride + c * span.in_channel_stride] = total;
                            }
                        }
                    }
                }
            }
        }
    }
}
