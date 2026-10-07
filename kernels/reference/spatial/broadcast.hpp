/**************************************************************************/
/* broadcast.hpp                                                          */
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
 * @file broadcast.hpp
 * @brief Defines the reference kernels of the three broadcast copies.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/support/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Applies a per-channel scale and shift, `out = in * gamma[c] + beta[c]`.
             * @param span The image layout; `out_h` and `out_w` equal `in_h` and `in_w`.
             * @param gamma_stride Elements between adjacent entries of @p gamma.
             * @param beta_stride Elements between adjacent entries of @p beta.
             * @see kernels/README.md
             */
            template<typename T>
            void channel_affine(const Spatial::WindowSpan& span, const T* in,
                                const T* gamma, const T* beta,
                                int64_t gamma_stride, int64_t beta_stride, T* out)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                const T value = in[n * span.in_batch_stride
                                                 + h * span.in_row_stride
                                                 + w * span.in_col_stride
                                                 + c * span.in_channel_stride];

                                out[n * span.out_batch_stride + h * span.out_row_stride
                                    + w * span.out_col_stride + c * span.out_channel_stride] =
                                    value * gamma[c * gamma_stride] + beta[c * beta_stride];
                            }
                        }
                    }
                }
            }

            /**
             * @brief Spreads an `[N, C]` value over `[N, H, W, C]`, scaled.
             * @param span The output layout; only the extents and the `out_*` strides are read.
             * @param in_batch_stride Elements between consecutive rows of @p in.
             * @param in_channel_stride Elements between consecutive entries of one row of @p in.
             * @see kernels/README.md
             */
            template<typename T>
            void broadcast_spatial(const Spatial::WindowSpan& span, const T* in,
                                   int64_t in_batch_stride, int64_t in_channel_stride,
                                   T* out, T scale)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.out_h; ++h)
                    {
                        for (int64_t w = 0; w < span.out_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                const T value = in[n * in_batch_stride + c * in_channel_stride];

                                out[n * span.out_batch_stride + h * span.out_row_stride
                                    + w * span.out_col_stride + c * span.out_channel_stride] =
                                    value * scale;
                            }
                        }
                    }
                }
            }

            /**
             * @brief Copies a rank-4 tensor element by logical `(n, c, h, w)` position.
             * @param span The two layouts; `out_h` equals `in_h` and `out_w` equals `in_w`.
             * @see kernels/README.md
             */
            template<typename T>
            void layout_convert(const Spatial::WindowSpan& span, const T* in, T* out)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                out[n * span.out_batch_stride + h * span.out_row_stride
                                    + w * span.out_col_stride + c * span.out_channel_stride] =
                                    in[n * span.in_batch_stride + h * span.in_row_stride
                                       + w * span.in_col_stride + c * span.in_channel_stride];
                            }
                        }
                    }
                }
            }
        }
    }
}
