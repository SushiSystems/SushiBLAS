/**************************************************************************/
/* bilinear_taps.hpp                                                      */
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
 * @file bilinear_taps.hpp
 * @brief Defines the taps, blend and weights that the reference and SYCL bilinear kernels share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Bilinear
        {
            /**
             * @brief Holds the two source indices and the blend weight of one output index.
             * @tparam A The type the weight is held in.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename A>
            struct Tap
            {
                /** @brief The lower source index, clamped to the extent. */
                int64_t low;
                /** @brief The upper source index, clamped to the extent. */
                int64_t high;
                /** @brief The share of the upper source: the value is `(1 - weight) * low + weight * high`. */
                A weight;
            };

            /**
             * @brief Returns the tap of output index `out_index` for an integer scale.
             * @param scale Must be positive.
             * @param extent The source extent along the axis; must be positive.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename A>
            [[nodiscard]] inline Tap<A> tap(int64_t out_index, int64_t scale, int64_t extent) noexcept
            {
                // s = (out_index + 0.5) / scale - 0.5 = numerator / denominator, held in integers.
                const int64_t numerator = 2 * out_index + 1 - scale;
                const int64_t denominator = 2 * scale;

                // The numerator exceeds -denominator, so a negative one floors to -1.
                const int64_t below = numerator < 0 ? -1 : numerator / denominator;
                const int64_t remainder = numerator - below * denominator;

                const int64_t last = extent - 1;
                const int64_t lower = below < 0 ? 0 : (below > last ? last : below);
                const int64_t upper = below + 1 > last ? last : (below + 1 < 0 ? 0 : below + 1);

                return Tap<A>{lower, upper,
                              static_cast<A>(remainder) / static_cast<A>(denominator)};
            }

            /**
             * @brief Returns `p * (1 - t) + q * t`, and exactly `p` when t is zero.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename A>
            [[nodiscard]] inline A blend(A p, A q, A t) noexcept
            {
                // A zero weight must not read q: an infinite q would turn p into NaN.
                return t == A(0) ? p : p * (A(1) - t) + q * t;
            }

            /**
             * @brief Returns the weight a source index receives from the output index a tap describes.
             * @return Zero when the tap reads neither of its sources from index; the sum of both
             *         taps when clamping makes them the same pixel.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename A>
            [[nodiscard]] inline A axis_weight(const Tap<A>& t, int64_t index) noexcept
            {
                A weight = A(0);
                if (t.low == index)
                    weight += A(1) - t.weight;
                if (t.high == index)
                    weight += t.weight;
                return weight;
            }

            /**
             * @brief Interpolates one value from the four pixels a pair of taps names.
             * @param origin The element of the image and channel at row 0, column 0.
             * @param row_stride Elements between rows; col_stride likewise between columns.
             * @see modules/kernels/spatial_kernels/README.md
             */
            template<typename T>
            [[nodiscard]] inline Accumulator<T> sample(const T* origin, int64_t row_stride,
                                                       int64_t col_stride,
                                                       const Tap<Accumulator<T>>& row,
                                                       const Tap<Accumulator<T>>& col) noexcept
            {
                using A = Accumulator<T>;

                const A a = A(origin[row.low * row_stride + col.low * col_stride]);
                const A b = A(origin[row.low * row_stride + col.high * col_stride]);
                const A c = A(origin[row.high * row_stride + col.low * col_stride]);
                const A d = A(origin[row.high * row_stride + col.high * col_stride]);

                const A top = blend(a, b, col.weight);
                const A bottom = blend(c, d, col.weight);
                return blend(top, bottom, row.weight);
            }

            /**
             * @brief Returns the first output index that can read source index `index`.
             * @see modules/kernels/spatial_kernels/README.md
             */
            [[nodiscard]] inline int64_t first_output(int64_t index, int64_t scale) noexcept
            {
                const int64_t first = (index - 1) * scale;
                return first < 0 ? 0 : first;
            }

            /**
             * @brief Returns one past the last output index that can read source index `index`.
             * @param out_extent The output extent along the axis.
             * @see modules/kernels/spatial_kernels/README.md
             */
            [[nodiscard]] inline int64_t end_output(int64_t index, int64_t scale,
                                                    int64_t out_extent) noexcept
            {
                const int64_t end = (index + 2) * scale;
                return end > out_extent ? out_extent : end;
            }
        } // namespace Bilinear
    } // namespace Kernels
} // namespace SushiBLAS
