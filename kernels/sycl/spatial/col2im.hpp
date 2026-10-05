/****************************************************************/
/* col2im.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file col2im.hpp
 * @brief Defines the SYCL kernel of col2im.
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
             * @brief Scatters an im2col matrix back onto an image by summing overlaps.
             * @param span The window layout; its `in_*` fields describe the image this produces.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event col2im(sycl::queue& q, const Spatial::WindowSpan& span,
                               const Spatial::ColumnSpan& cols,
                               const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0) return sycl::event();

                // The device closure captures values, never a host reference.
                const Spatial::ConvGeometry g = span.geometry;
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t row_stride = cols.row_stride;
                const int64_t element_stride = cols.element_stride;

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

                            // Channel is innermost, so a sub-group reads contiguous channels.
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

                                    const int64_t row = (n * out_h + h_o) * out_w + w_o;
                                    const int64_t patch = (r * g.kernel_w + s) * channels + c;

                                    total_value += in[row * row_stride + patch * element_stride];
                                }
                            }

                            out[n * in_batch_stride + h * in_row_stride
                                + w * in_col_stride + c * in_channel_stride] = total_value;
                        }
                    });
                });
            }
        }
    }
}
