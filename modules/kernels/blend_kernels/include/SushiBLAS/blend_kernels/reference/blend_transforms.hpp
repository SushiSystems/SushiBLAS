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

/**
 * @file blend_transforms.hpp
 * @brief Defines the reference kernel of the weighted blend of transforms applied to points.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blend_kernels/blend_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Writes `out[p] = sum_j weights[p][j] * transforms[indices[p][j]] * point[p]`.
             * @param components 3 (a position, w = 1) or 4 (homogeneous).
             * @param transform_rows 3 (last row `(0, 0, 0, 1)` implied) or 4, of `[T][rows][4]`.
             * @param transform_count T; an index outside `[0, T)` adds no term.
             * @param out Holds `components` values per point; every one is written.
             * @see kernels/README.md
             */
            template<typename T, typename Index>
            void blend_transforms(int64_t point_count, int64_t components, int64_t transform_rows,
                                  int64_t influences, int64_t transform_count, const T* points,
                                  const T* transforms, const Index* indices, const T* weights,
                                  T* out)
            {
                using RW = ReferenceAccumulator<T>;

                for (int64_t p = 0; p < point_count; ++p)
                {
                    const T* point = points + Indexing::point_offset(p, components);

                    // A position has no w, so it is the homogeneous point with w = 1.
                    const RW homogeneous[4] = {RW(point[0]), RW(point[1]), RW(point[2]),
                                               Indexing::homogeneous_w<RW>(point, components)};

                    for (int64_t i = 0; i < components; ++i)
                    {
                        // The influences are added in ascending j order; see kernels/README.md.
                        ReferenceSum<T> blended;

                        for (int64_t j = 0; j < influences; ++j)
                        {
                            const int64_t slot = static_cast<int64_t>(
                                indices[Indexing::influence_offset(p, influences, j)]);
                            if (!Indexing::slot_live(slot, transform_count))
                                continue;

                            const T* matrix =
                                transforms + Indexing::matrix_offset(slot, transform_rows);
                            RW moved = homogeneous[3];

                            if (Indexing::has_matrix_row(i, transform_rows))
                            {
                                ReferenceSum<T> row;
                                for (int64_t c = 0; c < 4; ++c)
                                    row.add(RW(matrix[Indexing::matrix_element(i, c)]) * homogeneous[c]);
                                moved = row.value();
                            }

                            blended.add(RW(weights[Indexing::influence_offset(p, influences, j)]) * moved);
                        }

                        out[Indexing::point_offset(p, components) + i] = static_cast<T>(blended.value());
                    }
                }
            }
        }
    }
}
