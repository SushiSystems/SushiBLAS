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
 * @brief Defines the SYCL kernels of nearest-neighbour upsampling and its gradient.
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
#include <SushiBLAS/spatial_kernels/upsample_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Repeats each input pixel into a `scale_h` by `scale_w` block.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event upsample_nearest(sycl::queue& q, const Spatial::WindowSpan& span,
                                         int64_t scale_h, int64_t scale_w,
                                         const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.output_elements();
                if (total <= 0)
                    return sycl::event();

                const int64_t channels = span.channels;
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

                            const int64_t h_in = Indexing::upsample_source(h_o, sh);
                            const int64_t w_in = Indexing::upsample_source(w_o, sw);

                            out[out_s.offset(n, h_o, w_o, c)] =
                                in[in_s.offset(n, h_in, w_in, c)];
                        }
                    });
                });
            }

            /**
             * @brief Sums each output block back onto the pixel that produced it.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event upsample_nearest_backward(sycl::queue& q, const Spatial::WindowSpan& span,
                                                  int64_t scale_h, int64_t scale_w,
                                                  const T* dy, T* dx,
                                                  const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0)
                    return sycl::event();

                const int64_t channels = span.channels;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
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

                            Accumulator<T> total_value = Accumulator<T>(0);

                            for (int64_t dh = 0; dh < sh; ++dh)
                            {
                                for (int64_t dw = 0; dw < sw; ++dw)
                                {
                                    total_value += Accumulator<T>(
                                        dy[out_s.offset(n, Indexing::upsample_block(h, sh, dh),
                                                     Indexing::upsample_block(w, sw, dw), c)]);
                                }
                            }

                            dx[in_s.offset(n, h, w, c)] =
                                static_cast<T>(total_value);
                        }
                    });
                });
            }
        }
    }
}
