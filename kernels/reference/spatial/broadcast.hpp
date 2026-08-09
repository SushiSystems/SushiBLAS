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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file broadcast.hpp
 * @brief Three broadcast copies: a per-channel affine, a per-(n, c) spread, and
 * a rank-4 permuting copy.
 *
 * **Nothing in this file accumulates.** Every one of these kernels writes one
 * output element per iteration, reading one input element and at most one
 * per-channel scalar; there is no running total, no reduction and no operand
 * whose combination order could differ between two runs or between the
 * reference and the device. That is worth stating rather than leaving to be
 * observed, because the rest of this domain's kernels *do* accumulate and are
 * written the way they are (gathers, never atomic scatters) precisely to keep
 * that order fixed. Here there is no order to fix, so the reference and the
 * SYCL twin agree bit for bit by construction and the test that compares them
 * is checking indexing arithmetic, not numerics.
 */

#include <cstdint>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Applies a per-channel scale and shift.
             *
             * `out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`. BatchNorm's
             * apply half, and the whole of BatchNorm in evaluation mode.
             *
             * @p gamma and @p beta are rank-1 `[C]` operands addressed through
             * their own strides rather than assumed contiguous, so a parameter
             * vector that is a view into a packed parameter arena works without
             * a compacting copy.
             *
             * @tparam T The element type.
             * @param span The image layout, in element counts; `out_h` and
             *        `out_w` equal `in_h` and `in_w`.
             * @param in The input image's first element.
             * @param gamma The scale vector's first element.
             * @param beta The shift vector's first element.
             * @param gamma_stride Elements between adjacent entries of @p gamma.
             * @param beta_stride Elements between adjacent entries of @p beta.
             * @param out The output image's first element.
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
             *
             * `out[n,h,w,c] = in[n,c] * scale`. Global average pooling's
             * backward at `scale = 1/(H*W)`.
             *
             * The input is rank-2, so its two strides are parameters rather than
             * fields of @p span: a @ref Spatial::WindowSpan has four input
             * strides and only two of them would mean anything here, and two
             * meaningless fields are two a reader has to rule out.
             *
             * @tparam T The element type.
             * @param span The output image's layout, in element counts; only the
             *        extents and the `out_*` strides are read.
             * @param in The rank-2 input's first element.
             * @param in_batch_stride Elements between consecutive rows of @p in.
             * @param in_channel_stride Elements between consecutive entries of
             *        one row of @p in.
             * @param out The output image's first element.
             * @param scale The factor every element is multiplied by.
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
             *
             * The boundary transpose between NCHW and NHWC. **Its direction is
             * not a flag**: both sides are indexed through their own four
             * strides, so the caller expresses which way the transpose runs by
             * handing in a differently-strided output, and one kernel serves
             * both directions.
             *
             * @tparam T The element type.
             * @param span The two layouts, in element counts; the logical
             *        extents are shared, so `out_h` equals `in_h` and `out_w`
             *        equals `in_w`.
             * @param in The input's first element.
             * @param out The output's first element.
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
