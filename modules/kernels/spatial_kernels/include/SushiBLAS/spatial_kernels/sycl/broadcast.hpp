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
 * @file broadcast.hpp
 * @brief Defines the SYCL kernels of the three broadcast copies.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/pixel_index.hpp>
#include <SushiBLAS/spatial_kernels/transpose_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Applies a per-channel scale and shift, one work-item per output element.
             * @param gamma_stride Elements between adjacent entries of @p gamma.
             * @param beta_stride Elements between adjacent entries of @p beta.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event channel_affine(sycl::queue& q, const Spatial::WindowSpan& span,
                                       const T* in, const T* gamma, const T* beta,
                                       int64_t gamma_stride, int64_t beta_stride, T* out,
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
                const int64_t g_stride = gamma_stride;
                const int64_t b_stride = beta_stride;

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

                            const T value = in[in_s.offset(n, h, w, c)];

                            out[out_s.offset(n, h, w, c)] =
                                value * gamma[c * g_stride] + beta[c * b_stride];
                        }
                    });
                });
            }

            /**
             * @brief Spreads an `[N, C]` value over `[N, H, W, C]`, scaled.
             * @param in_batch_stride Elements between consecutive rows of @p in.
             * @param in_channel_stride Elements between consecutive entries of one row of @p in.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event broadcast_spatial(sycl::queue& q, const Spatial::WindowSpan& span,
                                          const T* in, int64_t in_batch_stride,
                                          int64_t in_channel_stride, T* out, T scale,
                                          const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.output_elements();
                if (total <= 0)
                    return sycl::event();

                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t src_batch_stride = in_batch_stride;
                const int64_t src_channel_stride = in_channel_stride;
                const Indexing::Strides out_s = Indexing::output_strides(span);
                const T factor = scale;

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

                            const auto [n, h, w, c] = Indexing::pixel_coord(i, out_h, out_w, channels);

                            const T value = in[Indexing::batch_channel_offset(n, c, src_batch_stride, src_channel_stride)];

                            out[out_s.offset(n, h, w, c)] = value * factor;
                        }
                    });
                });
            }

            namespace Detail
            {
                /**
                 * @brief Copies every image through square local tiles, reading along the input's
                 *        adjacent axis and writing along the output's.
                 * @see kernels/README.md
                 */
                template<typename T>
                sycl::event transpose_images(sycl::queue& q, const Indexing::TransposePlan& plan,
                                             int64_t batch, const T* in, T* out,
                                             const std::vector<sycl::event>& deps)
                {
                    constexpr int K_TILE = Indexing::K_TRANSPOSE_TILE;
                    constexpr int K_ROWS = Indexing::K_TRANSPOSE_ROWS;
                    const Indexing::TransposePlan p = plan;
                    const std::size_t slow_tiles =
                        static_cast<std::size_t>(Indexing::tile_count(p.slow));
                    const std::size_t fast_tiles =
                        static_cast<std::size_t>(Indexing::tile_count(p.fast));
                    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                                slow_tiles * K_ROWS, fast_tiles * K_TILE);
                    const sycl::range<3> local(1, K_ROWS, K_TILE);

                    return q.submit([&](sycl::handler& handler)
                    {
                        handler.depends_on(deps);

                        // One padding column keeps a column of the tile off a single bank.
                        const std::size_t tile_elements =
                            static_cast<std::size_t>(Indexing::tile_elements());
                        sycl::local_accessor<T, 1> tile(sycl::range<1>(tile_elements), handler);

                        handler.parallel_for(sycl::nd_range<3>(global, local),
                                             [=](sycl::nd_item<3> it)
                        {
                            const int64_t n = static_cast<int64_t>(it.get_group(0));
                            const int64_t slow0 = static_cast<int64_t>(it.get_group(1)) * K_TILE;
                            const int64_t fast0 = static_cast<int64_t>(it.get_group(2)) * K_TILE;
                            const int ty = static_cast<int>(it.get_local_id(1));
                            const int tx = static_cast<int>(it.get_local_id(2));
                            const T* image_in = in + n * p.in_batch;
                            T* image_out = out + n * p.out_batch;

                            for (int r = ty; r < K_TILE; r += K_ROWS)
                            {
                                const int64_t s = slow0 + r;
                                const int64_t f = fast0 + tx;
                                if (Indexing::tile_live(p, s, f))
                                    tile[Indexing::tile_slot(r, tx)] =
                                        image_in[Indexing::transpose_read_offset(p, s, f)];
                            }
                            sycl::group_barrier(it.get_group());

                            for (int r = ty; r < K_TILE; r += K_ROWS)
                            {
                                const int64_t s = slow0 + tx;
                                const int64_t f = fast0 + r;
                                if (Indexing::tile_live(p, s, f))
                                    image_out[Indexing::transpose_write_offset(p, s, f)] =
                                        tile[Indexing::tile_slot(tx, r)];
                            }
                        });
                    });
                }
            } // namespace Detail

            /**
             * @brief Copies a rank-4 tensor element by logical `(n, c, h, w)` position.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event layout_convert(sycl::queue& q, const Spatial::WindowSpan& span,
                                       const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0)
                    return sycl::event();

                Indexing::TransposePlan plan;
                if (Indexing::transposes(span, plan))
                    return Detail::transpose_images(q, plan, span.batch, in, out, deps);

                const int64_t channels = span.channels;
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

                            out[out_s.offset(n, h, w, c)] =
                                in[in_s.offset(n, h, w, c)];
                        }
                    });
                });
            }
        }
    }
}
