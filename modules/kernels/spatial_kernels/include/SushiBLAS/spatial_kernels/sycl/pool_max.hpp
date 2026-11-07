/**************************************************************************/
/* pool_max.hpp                                                           */
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
 * @file pool_max.hpp
 * @brief Defines the SYCL kernels of max pooling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/window_index.hpp>
#include <SushiBLAS/spatial_kernels/pool_max_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Takes each window's maximum, or its first NaN, and records where it came from.
             * @param indices The argmax buffer's first element; laid out exactly like @p out.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event pool_max(sycl::queue& q, const Spatial::WindowSpan& span,
                                 const T* in, T* out, int64_t* indices,
                                 const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.output_elements();
                if (total <= 0)
                    return sycl::event();

                const Spatial::ConvGeometry g = span.geometry;
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);

                // lowest() is not constexpr for sycl::half, so this is a const.
                const T lowest = std::numeric_limits<T>::lowest();

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

                            T best = lowest;
                            int64_t best_index = -1;

                            for (int64_t r = 0; r < g.kernel_h; ++r)
                            {
                                const int64_t h_in = Indexing::gather_coord(h_o, g.stride_h, g.pad_top, r, g.dilation_h);
                                if (!Indexing::in_extent(h_in, in_h))
                                    continue;

                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t w_in = Indexing::gather_coord(w_o, g.stride_w, g.pad_left, s, g.dilation_w);
                                    if (!Indexing::in_extent(w_in, in_w))
                                        continue;

                                    const T candidate = in[in_s.offset(n, h_in, w_in, c)];

                                    // The first live tap is taken; a NaN tap wins and stays.
                                    if (Indexing::takes_candidate(best_index, best, candidate))
                                    {
                                        best = candidate;
                                        best_index = Indexing::flat_pixel({n, h_in, w_in, c}, in_h, in_w, channels);
                                    }
                                }
                            }

                            const int64_t at = out_s.offset(n, h_o, w_o, c);
                            out[at] = best;
                            indices[at] = best_index;
                        }
                    });
                });
            }

            /**
             * @brief Routes each output gradient back to the element it was taken from.
             * @param indices The argmax buffer `pool_max` wrote.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event pool_max_backward(sycl::queue& q, const Spatial::WindowSpan& span,
                                          const T* dy, const int64_t* indices, T* dx,
                                          const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0)
                    return sycl::event();

                const Spatial::ConvGeometry g = span.geometry;
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);

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

                            const int64_t mine = Indexing::flat_pixel({n, h, w, c}, in_h, in_w, channels);

                            Accumulator<T> total_value = Accumulator<T>(0);

                            for (int64_t r = 0; r < g.kernel_h; ++r)
                            {
                                const int64_t h_o = Indexing::scatter_coord(h, g.stride_h, g.pad_top, r, g.dilation_h, out_h);
                                if (h_o < 0)
                                    continue;

                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t w_o = Indexing::scatter_coord(w, g.stride_w, g.pad_left, s, g.dilation_w, out_w);
                                    if (w_o < 0)
                                        continue;

                                    const int64_t at = out_s.offset(n, h_o, w_o, c);

                                    if (indices[at] == mine)
                                        total_value += Accumulator<T>(dy[at]);
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
