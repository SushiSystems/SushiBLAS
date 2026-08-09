/**************************************************************************/
/* im2col.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             *
             * One work-item per *output* element, which is what makes this a pure
             * gather: every item writes exactly one entry and reads exactly one
             * input element (or none, for a padded tap). No two items write the
             * same address, so there is nothing to synchronise and nothing whose
             * combination order could vary between runs.
             *
             * Consecutive items differ in `c` first, so a sub-group reads a
             * contiguous run of channels - which is the NHWC half of the
             * argument for choosing that layout.
             *
             * @tparam T The element type.
             * @param q The queue to submit to.
             * @param span The window layout, in element counts.
             * @param cols The column matrix's addressing.
             * @param in The input's first element.
             * @param out The column matrix's first element.
             * @param deps Events this submission must depend on.
             * @return The event for the submitted kernel.
             */
            template<typename T>
            sycl::event im2col(sycl::queue& q, const Spatial::WindowSpan& span,
                               const Spatial::ColumnSpan& cols,
                               const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.column_elements();
                if (total <= 0) return sycl::event();

                // Copied out by value: the closure below is device code and may
                // not chase a host reference.
                const Spatial::ConvGeometry g = span.geometry;
                const int64_t patch_width = span.patch_elements();
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

                            const int64_t col = i % patch_width;
                            const int64_t row = i / patch_width;

                            const int64_t c = col % channels;
                            const int64_t rs = col / channels;
                            const int64_t s = rs % g.kernel_w;
                            const int64_t r = rs / g.kernel_w;

                            const int64_t w_o = row % out_w;
                            const int64_t nh = row / out_w;
                            const int64_t h_o = nh % out_h;
                            const int64_t n = nh / out_h;

                            const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                            const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;

                            const bool ok = h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w;

                            out[row * row_stride + col * element_stride] = ok
                                ? in[n * in_batch_stride + h_in * in_row_stride
                                     + w_in * in_col_stride + c * in_channel_stride]
                                : T(0);
                        }
                    });
                });
            }
        }
    }
}
