/**************************************************************************/
/* channels.hpp                                                           */
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
 * @file channels.hpp
 * @brief Defines the SYCL kernel that copies a run of channels.
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

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Copies a run of channels from one image into another.
             * @param count How many channels to copy; it is not either tensor's channel count.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event copy_channel_range(sycl::queue& q, const Spatial::WindowSpan& span,
                                           int64_t src_channel_offset, int64_t dst_channel_offset,
                                           int64_t count, const T* in, T* out,
                                           const std::vector<sycl::event>& deps)
            {
                if (count <= 0)
                    return sycl::event();

                const int64_t total = span.batch * span.in_h * span.in_w * count;
                if (total <= 0)
                    return sycl::event();

                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t width = count;
                const int64_t src_base = src_channel_offset;
                const int64_t dst_base = dst_channel_offset;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Indexing::Strides out_s = Indexing::output_strides(span);

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

                            // The channel slot of the run plays the part of the channel: width is its extent.
                            const auto [n, h, w, k] = Indexing::pixel_coord(i, in_h, in_w, width);

                            out[out_s.offset(n, h, w, dst_base + k)] =
                                in[in_s.offset(n, h, w, src_base + k)];
                        }
                    });
                });
            }
        }
    }
}
