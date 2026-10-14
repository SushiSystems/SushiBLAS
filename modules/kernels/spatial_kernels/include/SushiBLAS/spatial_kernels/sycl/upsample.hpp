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
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;
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

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w_o = hw % out_w;
                            const int64_t nh = hw / out_w;
                            const int64_t h_o = nh % out_h;
                            const int64_t n = nh / out_h;

                            const int64_t h_in = h_o / sh;
                            const int64_t w_in = w_o / sw;

                            out[n * out_batch_stride + h_o * out_row_stride
                                + w_o * out_col_stride + c * out_channel_stride] =
                                in[n * in_batch_stride + h_in * in_row_stride
                                   + w_in * in_col_stride + c * in_channel_stride];
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
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;
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

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % in_w;
                            const int64_t nh = hw / in_w;
                            const int64_t h = nh % in_h;
                            const int64_t n = nh / in_h;

                            Accumulator<T> total_value = Accumulator<T>(0);

                            for (int64_t dh = 0; dh < sh; ++dh)
                            {
                                for (int64_t dw = 0; dw < sw; ++dw)
                                {
                                    total_value += Accumulator<T>(
                                        dy[n * out_batch_stride
                                           + (h * sh + dh) * out_row_stride
                                           + (w * sw + dw) * out_col_stride
                                           + c * out_channel_stride]);
                                }
                            }

                            dx[n * in_batch_stride + h * in_row_stride
                               + w * in_col_stride + c * in_channel_stride] =
                                static_cast<T>(total_value);
                        }
                    });
                });
            }
        }
    }
}
