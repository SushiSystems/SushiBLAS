/**************************************************************************/
/* window_dot_weights.hpp                                                 */
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
 * @file window_dot_weights.hpp
 * @brief Defines the reference kernel of the weight gradient of the per-channel weighted window sum.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/support/spatial_geometry.hpp>
#include <SushiBLAS/support/reference_accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Sums, for each tap of each channel, every output gradient times the input element that tap read.
             * @param span The forward's layout; its `in_*` fields describe in and its `out_*` fields dy.
             * @param dweights The `[kernel_h, kernel_w, channels]` gradient; every element is written.
             * @see kernels/README.md
             */
            template<typename T>
            void window_dot_weights_backward(const Spatial::WindowSpan& span,
                                             const Spatial::TapSpan& taps, const T* dy,
                                             const T* in, T* dweights)
            {
                using RW = ReferenceAccumulator<T>;
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t r = 0; r < g.kernel_h; ++r)
                {
                    for (int64_t s = 0; s < g.kernel_w; ++s)
                    {
                        for (int64_t c = 0; c < span.channels; ++c)
                        {
                            // The terms are added in ascending (n, h_o, w_o) order; see kernels/README.md.
                            ReferenceSum<T> sum;

                            for (int64_t n = 0; n < span.batch; ++n)
                            {
                                for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                                {
                                    const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                    if (h_in < 0 || h_in >= span.in_h) continue;

                                    for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                                    {
                                        const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                        if (w_in < 0 || w_in >= span.in_w) continue;

                                        sum.add(RW(dy[n * span.out_batch_stride
                                                      + h_o * span.out_row_stride
                                                      + w_o * span.out_col_stride
                                                      + c * span.out_channel_stride])
                                                * RW(in[n * span.in_batch_stride
                                                        + h_in * span.in_row_stride
                                                        + w_in * span.in_col_stride
                                                        + c * span.in_channel_stride]));
                                    }
                                }
                            }

                            dweights[r * taps.row_stride + s * taps.col_stride
                                     + c * taps.channel_stride] = static_cast<T>(sum.value());
                        }
                    }
                }
            }
        }
    }
}
