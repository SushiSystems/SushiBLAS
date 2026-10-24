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
 * @brief Defines the reference kernel of the weighted blend of transforms applied to points.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/reference/reference_accumulator.hpp>

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
                    const T* point = points + p * components;

                    // A position has no w, so it is the homogeneous point with w = 1.
                    const RW homogeneous[4] = {RW(point[0]), RW(point[1]), RW(point[2]),
                                               components == 4 ? RW(point[3]) : RW(1)};

                    for (int64_t i = 0; i < components; ++i)
                    {
                        // The influences are added in ascending j order; see kernels/README.md.
                        ReferenceSum<T> blended;

                        for (int64_t j = 0; j < influences; ++j)
                        {
                            const int64_t slot = static_cast<int64_t>(indices[p * influences + j]);
                            if (slot < 0 || slot >= transform_count)
                                continue;

                            const T* matrix = transforms + slot * transform_rows * 4;
                            RW moved = homogeneous[3];

                            if (i < transform_rows)
                            {
                                ReferenceSum<T> row;
                                for (int64_t c = 0; c < 4; ++c)
                                    row.add(RW(matrix[i * 4 + c]) * homogeneous[c]);
                                moved = row.value();
                            }

                            blended.add(RW(weights[p * influences + j]) * moved);
                        }

                        out[p * components + i] = static_cast<T>(blended.value());
                    }
                }
            }
        }
    }
}
