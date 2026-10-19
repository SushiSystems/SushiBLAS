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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;
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

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % in_w;
                            const int64_t nh = hw / in_w;
                            const int64_t h = nh % in_h;
                            const int64_t n = nh / in_h;

                            const T value = in[n * in_batch_stride + h * in_row_stride
                                             + w * in_col_stride + c * in_channel_stride];

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride + c * out_channel_stride] =
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
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;
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

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % out_w;
                            const int64_t nh = hw / out_w;
                            const int64_t h = nh % out_h;
                            const int64_t n = nh / out_h;

                            const T value = in[n * src_batch_stride + c * src_channel_stride];

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride + c * out_channel_stride] = value * factor;
                        }
                    });
                });
            }

            namespace Detail
            {
                /** @brief Holds the side of one square tile of the transposing copy. */
                inline constexpr int K_TRANSPOSE_TILE = 32;

                /** @brief Holds the rows of work-items of one transposing work-group. */
                inline constexpr int K_TRANSPOSE_ROWS = 8;

                /**
                 * @brief Holds one image as a matrix whose fast axis is adjacent in the input and
                 *        whose slow axis is adjacent in the output.
                 */
                struct TransposePlan
                {
                    int64_t fast = 0;       /**< Extent of the axis adjacent in the input. */
                    int64_t slow = 0;       /**< Extent of the axis adjacent in the output. */
                    int64_t in_slow = 0;    /**< Input step along the slow axis. */
                    int64_t out_fast = 0;   /**< Output step along the fast axis. */
                    int64_t in_batch = 0;   /**< Input step from one image to the next. */
                    int64_t out_batch = 0;  /**< Output step from one image to the next. */
                };

                /**
                 * @brief Returns whether a conversion moves the adjacent axis between pixels and
                 *        channels over linear image planes, and fills plan when it does.
                 * @see kernels/README.md
                 */
                [[nodiscard]] inline bool transposes(const Spatial::WindowSpan& span,
                                                     TransposePlan& plan) noexcept
                {
                    const bool planes = span.in_row_stride == span.in_w * span.in_col_stride &&
                                        span.out_row_stride == span.in_w * span.out_col_stride;
                    const int64_t pixels = span.in_h * span.in_w;
                    plan.in_batch = span.in_batch_stride;
                    plan.out_batch = span.out_batch_stride;
                    if (planes && span.in_channel_stride == 1 && span.out_col_stride == 1)
                    {
                        plan.fast = span.channels;
                        plan.slow = pixels;
                        plan.in_slow = span.in_col_stride;
                        plan.out_fast = span.out_channel_stride;
                        return true;
                    }
                    if (planes && span.in_col_stride == 1 && span.out_channel_stride == 1)
                    {
                        plan.fast = pixels;
                        plan.slow = span.channels;
                        plan.in_slow = span.in_channel_stride;
                        plan.out_fast = span.out_col_stride;
                        return true;
                    }
                    return false;
                }

                /**
                 * @brief Copies every image through square local tiles, reading along the input's
                 *        adjacent axis and writing along the output's.
                 * @see kernels/README.md
                 */
                template<typename T>
                sycl::event transpose_images(sycl::queue& q, const TransposePlan& plan,
                                             int64_t batch, const T* in, T* out,
                                             const std::vector<sycl::event>& deps)
                {
                    constexpr int K_TILE = K_TRANSPOSE_TILE;
                    constexpr int K_ROWS = K_TRANSPOSE_ROWS;
                    const TransposePlan p = plan;
                    const std::size_t slow_tiles =
                        static_cast<std::size_t>((p.slow + K_TILE - 1) / K_TILE);
                    const std::size_t fast_tiles =
                        static_cast<std::size_t>((p.fast + K_TILE - 1) / K_TILE);
                    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                                slow_tiles * K_ROWS, fast_tiles * K_TILE);
                    const sycl::range<3> local(1, K_ROWS, K_TILE);

                    return q.submit([&](sycl::handler& handler)
                    {
                        handler.depends_on(deps);

                        // One padding column keeps a column of the tile off a single bank.
                        const std::size_t tile_elements =
                            static_cast<std::size_t>(K_TILE * (K_TILE + 1));
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
                                if (s < p.slow && f < p.fast)
                                    tile[r * (K_TILE + 1) + tx] = image_in[s * p.in_slow + f];
                            }
                            sycl::group_barrier(it.get_group());

                            for (int r = ty; r < K_TILE; r += K_ROWS)
                            {
                                const int64_t s = slow0 + tx;
                                const int64_t f = fast0 + r;
                                if (s < p.slow && f < p.fast)
                                    image_out[s + f * p.out_fast] = tile[tx * (K_TILE + 1) + r];
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

                Detail::TransposePlan plan;
                if (Detail::transposes(span, plan))
                    return Detail::transpose_images(q, plan, span.batch, in, out, deps);

                const int64_t channels = span.channels;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;

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

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % in_w;
                            const int64_t nh = hw / in_w;
                            const int64_t h = nh % in_h;
                            const int64_t n = nh / in_h;

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride + c * out_channel_stride] =
                                in[n * in_batch_stride + h * in_row_stride
                                   + w * in_col_stride + c * in_channel_stride];
                        }
                    });
                });
            }
        }
    }
}
