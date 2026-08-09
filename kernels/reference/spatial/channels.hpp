/**************************************************************************/
/* channels.hpp                                                           */
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
             * @brief Copies a run of channels from one image into another.
             *
             * The whole of channel concatenation and channel slicing (design
             * decision D-A6): concatenation is this copy with the source's
             * offset at zero, slicing is the same copy with the destination's
             * at zero. Nothing distinguishes the two directions once the offsets
             * are named, so there is one kernel and two public names.
             *
             * @p span carries the shared `[N, H, W]` grid and **both** operands'
             * strides - `in_*` describing the source, `out_*` the destination.
             * `span.channels` is not read: the copy's width is @p count, which
             * is a different number from either tensor's channel count and has
             * to be passed rather than inferred. `span.out_h` and `span.out_w`
             * hold the input extents, because a copy has no window and therefore
             * no output extent distinct from its input's.
             *
             * @tparam T The element type.
             * @param span The grid and both operands' strides, in element counts.
             * @param src_channel_offset The first source channel to read.
             * @param dst_channel_offset The first destination channel to write.
             * @param count How many channels to copy.
             * @param in The source image's first element.
             * @param out The destination image's first element.
             */
            template<typename T>
            void copy_channel_range(const Spatial::WindowSpan& span, int64_t src_channel_offset,
                                    int64_t dst_channel_offset, int64_t count,
                                    const T* in, T* out)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t k = 0; k < count; ++k)
                            {
                                out[n * span.out_batch_stride
                                    + h * span.out_row_stride
                                    + w * span.out_col_stride
                                    + (dst_channel_offset + k) * span.out_channel_stride] =
                                    in[n * span.in_batch_stride
                                       + h * span.in_row_stride
                                       + w * span.in_col_stride
                                       + (src_channel_offset + k) * span.in_channel_stride];
                            }
                        }
                    }
                }
            }
        }
    }
}
