/****************************************************************/
/* channels.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
             * @brief Copies a run of channels from one image into another.
             *
             * One work-item per copied element, `N * H * W * count` of them,
             * with the copied-channel index innermost so an NHWC copy of a
             * contiguous range reads and writes consecutive addresses. See the
             * reference kernel of the same name for why one kernel serves both
             * concatenation and slicing, and for what `span` carries.
             *
             * @tparam T The element type.
             * @param q The queue to submit to.
             * @param span The grid and both operands' strides, in element counts.
             * @param src_channel_offset The first source channel to read.
             * @param dst_channel_offset The first destination channel to write.
             * @param count How many channels to copy.
             * @param in The source image's first element.
             * @param out The destination image's first element.
             * @param deps Events this submission must depend on.
             * @return The event for the submitted kernel.
             */
            template<typename T>
            sycl::event copy_channel_range(sycl::queue& q, const Spatial::WindowSpan& span,
                                           int64_t src_channel_offset, int64_t dst_channel_offset,
                                           int64_t count, const T* in, T* out,
                                           const std::vector<sycl::event>& deps)
            {
                if (count <= 0) return sycl::event();

                const int64_t total = span.batch * span.in_h * span.in_w * count;
                if (total <= 0) return sycl::event();

                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t width = count;
                const int64_t src_base = src_channel_offset;
                const int64_t dst_base = dst_channel_offset;
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
                const std::size_t element_count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0);
                             flat < element_count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const int64_t k = i % width;
                            const int64_t hw = i / width;
                            const int64_t w = hw % in_w;
                            const int64_t nh = hw / in_w;
                            const int64_t h = nh % in_h;
                            const int64_t n = nh / in_h;

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride
                                + (dst_base + k) * out_channel_stride] =
                                in[n * in_batch_stride + h * in_row_stride
                                   + w * in_col_stride
                                   + (src_base + k) * in_channel_stride];
                        }
                    });
                });
            }
        }
    }
}
