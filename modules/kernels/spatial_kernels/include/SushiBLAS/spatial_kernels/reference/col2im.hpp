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
 * @file col2im.hpp
 * @brief Defines the reference kernel of col2im.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/window_index.hpp>

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
                                    const int64_t h_o = Indexing::scatter_coord(h, g.stride_h, g.pad_top, r, g.dilation_h, span.out_h);
                                    if (h_o < 0)
                                        continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_o = Indexing::scatter_coord(w, g.stride_w, g.pad_left, s, g.dilation_w, span.out_w);
                                        if (w_o < 0)
                                            continue;

                                        const int64_t row = Indexing::window_row({n, h_o, w_o}, span.out_h, span.out_w);
                                        const int64_t patch = Indexing::patch_column(
                                            {r, s, c}, span.channels, g.kernel_w);

                                        total += Accumulator<T>(
                                            in[Indexing::column_offset(cols, row, patch)]);
                                    }
                                }

                                out[Indexing::input_strides(span).offset(n, h, w, c)] =
                                    static_cast<T>(total);
                            }
                        }
                    }
                }
            }
        }
    }
}
