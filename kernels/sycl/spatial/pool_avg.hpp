/****************************************************************/
/* pool_avg.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file pool_avg.hpp
 * @brief Defines the SYCL kernels of average pooling and its gradient.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Averages each window, dividing by the window's area.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event pool_avg(sycl::queue& q, const Spatial::WindowSpan& span,
                                 const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.output_elements();
                if (total <= 0) return sycl::event();

                const Spatial::ConvGeometry g = span.geometry;
                const T area = static_cast<T>(g.kernel_h * g.kernel_w);
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
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
                            const int64_t w_o = hw % out_w;
                            const int64_t nh = hw / out_w;
                            const int64_t h_o = nh % out_h;
                            const int64_t n = nh / out_h;

                            T total_value = T(0);

                            for (int64_t r = 0; r < g.kernel_h; ++r)
                            {
                                const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                if (h_in < 0 || h_in >= in_h) continue;

                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                    if (w_in < 0 || w_in >= in_w) continue;

                                    total_value += in[n * in_batch_stride + h_in * in_row_stride
                                                    + w_in * in_col_stride + c * in_channel_stride];
                                }
                            }

                            out[n * out_batch_stride + h_o * out_row_stride
                                + w_o * out_col_stride + c * out_channel_stride] = total_value / area;
                        }
                    });
                });
            }

            /**
             * @brief Spreads each output gradient evenly over the window it came from.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event pool_avg_backward(sycl::queue& q, const Spatial::WindowSpan& span,
                                          const T* dy, T* dx, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0) return sycl::event();

                const Spatial::ConvGeometry g = span.geometry;
                const T area = static_cast<T>(g.kernel_h * g.kernel_w);
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
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

                            T total_value = T(0);

                            for (int64_t r = 0; r < g.kernel_h; ++r)
                            {
                                const int64_t h_num = h + g.pad_top - r * g.dilation_h;
                                if (h_num < 0 || h_num % g.stride_h != 0) continue;
                                const int64_t h_o = h_num / g.stride_h;
                                if (h_o >= out_h) continue;

                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t w_num = w + g.pad_left - s * g.dilation_w;
                                    if (w_num < 0 || w_num % g.stride_w != 0) continue;
                                    const int64_t w_o = w_num / g.stride_w;
                                    if (w_o >= out_w) continue;

                                    total_value += dy[n * out_batch_stride + h_o * out_row_stride
                                                    + w_o * out_col_stride
                                                    + c * out_channel_stride] / area;
                                }
                            }

                            dx[n * in_batch_stride + h * in_row_stride
                               + w * in_col_stride + c * in_channel_stride] = total_value;
                        }
                    });
                });
            }
        }
    }
}
