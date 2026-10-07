/**************************************************************************/
/* pool_avg.hpp                                                           */
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
 * @file pool_avg.hpp
 * @brief Defines the reference kernels of average pooling and its gradient.
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
             * @brief Averages each window, dividing by the window's area.
             * @see kernels/README.md
             */
            template<typename T>
            void pool_avg(const Spatial::WindowSpan& span, const T* in, T* out)
            {
                const Spatial::ConvGeometry& g = span.geometry;
                const Accumulator<T> area = static_cast<Accumulator<T>>(g.kernel_h * g.kernel_w);

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                Accumulator<T> total = Accumulator<T>(0);

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                    if (h_in < 0 || h_in >= span.in_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                        if (w_in < 0 || w_in >= span.in_w) continue;

                                        total += Accumulator<T>(
                                            in[n * span.in_batch_stride
                                               + h_in * span.in_row_stride
                                               + w_in * span.in_col_stride
                                               + c * span.in_channel_stride]);
                                    }
                                }

                                out[n * span.out_batch_stride + h_o * span.out_row_stride
                                    + w_o * span.out_col_stride + c * span.out_channel_stride] =
                                    static_cast<T>(total / area);
                            }
                        }
                    }
                }
            }

            /**
             * @brief Spreads each output gradient evenly over the window it came from.
             * @param dx The input gradient; every pixel is written, so it needs no prior clear.
             * @see kernels/README.md
             */
            template<typename T>
            void pool_avg_backward(const Spatial::WindowSpan& span, const T* dy, T* dx)
            {
                const Spatial::ConvGeometry& g = span.geometry;
                const Accumulator<T> area = static_cast<Accumulator<T>>(g.kernel_h * g.kernel_w);

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
                                    if (h_num < 0 || h_num % g.stride_h != 0) continue;
                                    const int64_t h_o = h_num / g.stride_h;
                                    if (h_o >= span.out_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_num = w + g.pad_left - s * g.dilation_w;
                                        if (w_num < 0 || w_num % g.stride_w != 0) continue;
                                        const int64_t w_o = w_num / g.stride_w;
                                        if (w_o >= span.out_w) continue;

                                        total += Accumulator<T>(
                                            dy[n * span.out_batch_stride
                                               + h_o * span.out_row_stride
                                               + w_o * span.out_col_stride
                                               + c * span.out_channel_stride]) / area;
                                    }
                                }

                                dx[n * span.in_batch_stride + h * span.in_row_stride
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
