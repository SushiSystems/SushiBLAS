/**************************************************************************/
/* blend_transforms.hpp                                                   */
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

/**
 * @file blend_transforms.hpp
 * @brief Defines the SYCL kernel of the weighted blend of transforms applied to points.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Writes `out[p] = sum_j weights[p][j] * transforms[indices[p][j]] * point[p]`.
             * @param components 3 (a position, w = 1) or 4 (homogeneous).
             * @param transform_rows 3 (last row `(0, 0, 0, 1)` implied) or 4, of `[T][rows][4]`.
             * @param transform_count T; an index outside `[0, T)` adds no term and reads nothing.
             * @param out Holds `components` values per point; every one is written.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            sycl::event blend_transforms(sycl::queue& q, int64_t point_count, int64_t components,
                                         int64_t transform_rows, int64_t influences,
                                         int64_t transform_count, const T* points,
                                         const T* transforms, const Index* indices,
                                         const T* weights, T* out,
                                         const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                if (point_count <= 0)
                    return sycl::event();

                const std::size_t count = static_cast<std::size_t>(point_count);
                const Device::Grid1D grid = Device::plan_1d(Device::profile_for(q), count);
                const std::size_t stride = grid.stride();

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t p = static_cast<int64_t>(flat);
                            const T* point = points + p * components;

                            // A position has no w, so it is the homogeneous point with w = 1.
                            const W homogeneous[4] = {W(point[0]), W(point[1]), W(point[2]),
                                                      components == 4 ? W(point[3]) : W(1)};
                            W blended[4] = {W(0), W(0), W(0), W(0)};

                            // The influences are added in ascending j order on every device.
                            for (int64_t j = 0; j < influences; ++j)
                            {
                                const int64_t slot =
                                    static_cast<int64_t>(indices[p * influences + j]);
                                if (slot < 0 || slot >= transform_count)
                                    continue;

                                const T* matrix = transforms + slot * transform_rows * 4;
                                const W weight = W(weights[p * influences + j]);

                                for (int64_t i = 0; i < components; ++i)
                                {
                                    W moved = homogeneous[3];

                                    if (i < transform_rows)
                                    {
                                        moved = W(0);
                                        for (int64_t c = 0; c < 4; ++c)
                                            moved += W(matrix[i * 4 + c]) * homogeneous[c];
                                    }

                                    blended[i] += weight * moved;
                                }
                            }

                            for (int64_t i = 0; i < components; ++i)
                                out[p * components + i] = static_cast<T>(blended[i]);
                        }
                    });
                });
            }
        }
    }
}
