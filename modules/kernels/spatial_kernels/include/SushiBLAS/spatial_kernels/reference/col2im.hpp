/**************************************************************************/
/* col2im.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file col2im.hpp
 * @brief Defines the reference kernel of col2im.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Scatters an im2col matrix back onto an image by summing overlaps.
             * @param span The window layout; its `in_*` fields describe the image this produces.
             * @see kernels/README.md
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
                                Accumulator<T> total = Accumulator<T>(0);

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_num = h + g.pad_top - r * g.dilation_h;
                                    if (h_num < 0 || h_num % g.stride_h != 0)
                                        continue;
                                    const int64_t h_o = h_num / g.stride_h;
                                    if (h_o >= span.out_h)
                                        continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_num = w + g.pad_left - s * g.dilation_w;
                                        if (w_num < 0 || w_num % g.stride_w != 0)
                                            continue;
                                        const int64_t w_o = w_num / g.stride_w;
                                        if (w_o >= span.out_w)
                                            continue;

                                        const int64_t row = (n * span.out_h + h_o) * span.out_w + w_o;
                                        const int64_t patch = (r * g.kernel_w + s) * span.channels + c;

                                        total += Accumulator<T>(
                                            in[row * cols.row_stride
                                               + patch * cols.element_stride]);
                                    }
                                }

                                out[n * span.in_batch_stride + h * span.in_row_stride
                                    + w * span.in_col_stride + c * span.in_channel_stride] =
                                    static_cast<T>(total);
                            }
                        }
                    }
                }
            }
        }
    }
}
