/****************************************************************/
/* im2col.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file im2col.hpp
 * @brief Defines the reference kernel of im2col.
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
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             * @see kernels/README.md
             */
            template<typename T>
            void im2col(const Spatial::WindowSpan& span, const Spatial::ColumnSpan& cols,
                        const T* in, T* out)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            const int64_t row = (n * span.out_h + h_o) * span.out_w + w_o;

                            for (int64_t r = 0; r < g.kernel_h; ++r)
                            {
                                const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                const bool row_ok = h_in >= 0 && h_in < span.in_h;

                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                    const bool ok = row_ok && w_in >= 0 && w_in < span.in_w;

                                    const int64_t patch = (r * g.kernel_w + s) * span.channels;
                                    const int64_t base = n * span.in_batch_stride
                                                       + h_in * span.in_row_stride
                                                       + w_in * span.in_col_stride;

                                    for (int64_t c = 0; c < span.channels; ++c)
                                    {
                                        out[row * cols.row_stride + (patch + c) * cols.element_stride] =
                                            ok ? in[base + c * span.in_channel_stride] : T(0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
