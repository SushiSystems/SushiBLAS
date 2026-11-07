/**************************************************************************/
/* upsample.hpp                                                           */
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
 * @file upsample.hpp
 * @brief Defines the reference kernels of nearest-neighbour upsampling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/upsample_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Repeats each input pixel into a `scale_h` by `scale_w` block.
             * @see kernels/README.md
             */
            template<typename T>
            void upsample_nearest(const Spatial::WindowSpan& span,
                                  int64_t scale_h, int64_t scale_w,
                                  const T* in, T* out)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        const int64_t h_in = Indexing::upsample_source(h_o, scale_h);

                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            const int64_t w_in = Indexing::upsample_source(w_o, scale_w);

                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                out[Indexing::output_strides(span).offset(n, h_o, w_o, c)] =
                                    in[Indexing::input_strides(span).offset(n, h_in, w_in, c)];
                            }
                        }
                    }
                }
            }

            /**
             * @brief Sums each output block back onto the pixel that produced it.
             * @param dx The input gradient; every pixel is written, so it needs no prior clear.
             * @see kernels/README.md
             */
            template<typename T>
            void upsample_nearest_backward(const Spatial::WindowSpan& span,
                                           int64_t scale_h, int64_t scale_w,
                                           const T* dy, T* dx)
            {
                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                Accumulator<T> total = Accumulator<T>(0);

                                for (int64_t dh = 0; dh < scale_h; ++dh)
                                {
                                    for (int64_t dw = 0; dw < scale_w; ++dw)
                                    {
                                        total += Accumulator<T>(
                                            dy[Indexing::output_strides(span).offset(n, (h * scale_h + dh), (w * scale_w + dw), c)]);
                                    }
                                }

                                dx[Indexing::input_strides(span).offset(n, h, w, c)] =
                                    static_cast<T>(total);
                            }
                        }
                    }
                }
            }
        }
    }
}
