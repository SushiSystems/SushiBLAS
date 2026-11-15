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
 * @brief Defines the SYCL kernels of bilinear upsampling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/bilinear_taps.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Interpolates each output pixel from the four input pixels around its centre.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename T>
            sycl::event upsample_bilinear(sycl::queue& q, const Spatial::WindowSpan& span,
                                          int64_t scale_h, int64_t scale_w,
                                          const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                using A = Accumulator<T>;

                const int64_t total = span.output_elements();
                if (total <= 0)
                    return sycl::event();

                const int64_t channels = span.channels;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);
                const int64_t sh = scale_h;
                const int64_t sw = scale_w;

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();
                const std::size_t count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const auto [n, h_o, w_o, c] = Indexing::pixel_coord(i, out_h, out_w, channels);

                            const A value = Bilinear::sample<T>(
                                in + in_s.offset(n, 0, 0, c), in_s.row, in_s.col,
                                Bilinear::tap<A>(h_o, sh, in_h), Bilinear::tap<A>(w_o, sw, in_w));

                            out[out_s.offset(n, h_o, w_o, c)] = static_cast<T>(value);
                        }
                    });
                });
            }

            /**
             * @brief Sums onto each input pixel the output gradients it contributed to, by its weights.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename T>
            sycl::event upsample_bilinear_backward(sycl::queue& q, const Spatial::WindowSpan& span,
                                                   int64_t scale_h, int64_t scale_w,
                                                   const T* dy, T* dx,
                                                   const std::vector<sycl::event>& deps)
            {
                using A = Accumulator<T>;

                const int64_t total = span.input_elements();
                if (total <= 0)
                    return sycl::event();

                const int64_t channels = span.channels;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);
                const int64_t sh = scale_h;
                const int64_t sw = scale_w;

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();
                const std::size_t count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const auto [n, h, w, c] = Indexing::pixel_coord(i, in_h, in_w, channels);

                            const int64_t h_end = Bilinear::end_output(h, sh, out_h);
                            const int64_t w_end = Bilinear::end_output(w, sw, out_w);

                            Accumulator<T> total_value = Accumulator<T>(0);

                            for (int64_t h_o = Bilinear::first_output(h, sh); h_o < h_end; ++h_o)
                            {
                                const A row_weight = Bilinear::axis_weight(
                                    Bilinear::tap<A>(h_o, sh, in_h), h);
                                if (row_weight == A(0))
                                    continue;

                                for (int64_t w_o = Bilinear::first_output(w, sw);
                                     w_o < w_end; ++w_o)
                                {
                                    const A col_weight = Bilinear::axis_weight(
                                        Bilinear::tap<A>(w_o, sw, in_w), w);
                                    if (col_weight == A(0))
                                        continue;

                                    total_value += A(dy[out_s.offset(n, h_o, w_o, c)])
                                                   * (row_weight * col_weight);
                                }
                            }

                            dx[in_s.offset(n, h, w, c)] = static_cast<T>(total_value);
                        }
                    });
                });
            }
        }
    }
}
