/**************************************************************************/
/* im2col.hpp                                                             */
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
             *
             * This is `im2col`'s specification, and the answer
             * `ExecutionMode::FORCE_REFERENCE` produces. Written as the obvious
             * nest of loops, in the obvious order, because a reference kernel
             * that is clever is a second implementation to be wrong rather than
             * a ground truth.
             *
             * Note that `ExecutionMode::VERIFY` does *not* compare the two
             * kernels despite what its name suggests - it runs the device kernel
             * and then overwrites the result with this one. The comparison lives
             * in `tests/functional/unit/test_spatial_im2col.cpp`, which runs the
             * two forced modes into two buffers and checks them bit for bit.
             *
             * Out-of-bounds taps write zero. Padding never materialises as a
             * copy of the input, which is what saves a full-tensor pass here and
             * a crop in `col2im`.
             *
             * @tparam T The element type.
             * @param span The window layout, in element counts.
             * @param cols The column matrix's addressing.
             * @param in The input's first element.
             * @param out The column matrix's first element.
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
