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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file channels.hpp
 * @brief Defines the reference kernel that copies a run of channels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Copies a run of channels from one image into another.
             * @param span The grid and both operands' strides; `span.channels` is not read.
             * @param count How many channels to copy; it is not either tensor's channel count.
             * @see kernels/README.md
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
                                out[Indexing::output_strides(span).offset(n, h, w, dst_channel_offset + k)] =
                                    in[Indexing::input_strides(span).offset(n, h, w, src_channel_offset + k)];
                            }
                        }
                    }
                }
            }
        }
    }
}
