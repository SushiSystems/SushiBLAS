/**************************************************************************/
/* upsample_bilinear.hpp                                                  */
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
 * @file upsample_bilinear.hpp
 * @brief Defines the reference kernels of bilinear upsampling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/bilinear_taps.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Interpolates each output pixel from the four input pixels around its centre.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename T>
            void upsample_bilinear(const Spatial::WindowSpan& span,
                                   int64_t scale_h, int64_t scale_w,
                                   const T* in, T* out)
            {
                using A = Accumulator<T>;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        const Bilinear::Tap<A> row = Bilinear::tap<A>(h_o, scale_h, span.in_h);

                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            const Bilinear::Tap<A> col = Bilinear::tap<A>(w_o, scale_w, span.in_w);

                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                const A value = Bilinear::sample<T>(
                                    in + in_s.offset(n, 0, 0, c), in_s.row, in_s.col, row, col);

                                out[out_s.offset(n, h_o, w_o, c)] = static_cast<T>(value);
                            }
                        }
                    }
                }
            }

            /**
             * @brief Sums onto each input pixel the output gradients it contributed to, by its weights.
             * @param dx The input gradient; every pixel is written, so it needs no prior clear.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename T>
            void upsample_bilinear_backward(const Spatial::WindowSpan& span,
                                            int64_t scale_h, int64_t scale_w,
                                            const T* dy, T* dx)
            {
                using RW = ReferenceAccumulator<T>;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h = 0; h < span.in_h; ++h)
                    {
                        const int64_t h_end = Bilinear::end_output(h, scale_h, span.out_h);

                        for (int64_t w = 0; w < span.in_w; ++w)
                        {
                            const int64_t w_end = Bilinear::end_output(w, scale_w, span.out_w);

                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                // The terms are added in ascending (h_o, w_o) order; see kernels/README.md.
                                ReferenceSum<T> sum;

                                for (int64_t h_o = Bilinear::first_output(h, scale_h);
                                     h_o < h_end; ++h_o)
                                {
                                    const RW row_weight = Bilinear::axis_weight(
                                        Bilinear::tap<RW>(h_o, scale_h, span.in_h), h);
                                    if (row_weight == RW(0))
                                        continue;

                                    for (int64_t w_o = Bilinear::first_output(w, scale_w);
                                         w_o < w_end; ++w_o)
                                    {
                                        const RW col_weight = Bilinear::axis_weight(
                                            Bilinear::tap<RW>(w_o, scale_w, span.in_w), w);
                                        if (col_weight == RW(0))
                                            continue;

                                        sum.add(RW(dy[out_s.offset(n, h_o, w_o, c)])
                                                * (row_weight * col_weight));
                                    }
                                }

                                dx[in_s.offset(n, h, w, c)] = static_cast<T>(sum.value());
                            }
                        }
                    }
                }
            }
        }
    }
}
