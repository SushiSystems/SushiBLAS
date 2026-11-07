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
 * @file im2col.hpp
 * @brief Defines the SYCL kernel of im2col.
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
#include <SushiBLAS/spatial_kernels/window_index.hpp>
#include <SushiBLAS/spatial_kernels/im2col_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             * @see kernels/README.md
             */
            template<typename T>
            sycl::event im2col(sycl::queue& q, const Spatial::WindowSpan& span,
                               const Spatial::ColumnSpan& cols,
                               const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.column_elements();
                if (total <= 0)
                    return sycl::event();

                // The device closure captures values, never a host reference.
                const Spatial::ConvGeometry g = span.geometry;
                const int64_t patch_width = span.patch_elements();
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const Indexing::Strides in_s = Indexing::input_strides(span);
                const Spatial::ColumnSpan cols_span = cols;

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

                            const Indexing::Im2colCoord at = Indexing::im2col_coord(i, patch_width, channels,
                                                                                    g.kernel_w, out_h, out_w);

                            const int64_t h_in = Indexing::gather_coord(
                                at.window.h_o, g.stride_h, g.pad_top, at.tap.r, g.dilation_h);
                            const int64_t w_in = Indexing::gather_coord(
                                at.window.w_o, g.stride_w, g.pad_left, at.tap.s, g.dilation_w);

                            const bool ok = Indexing::in_extent(h_in, in_h) &&
                                            Indexing::in_extent(w_in, in_w);

                            out[Indexing::column_offset(cols_span, at.row, at.col)] = ok
                                ? in[in_s.offset(at.window.n, h_in, w_in, at.tap.c)]
                                : T(0);
                        }
                    });
                });
            }
        }
    }
}
