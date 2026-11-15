/**************************************************************************/
/* test_upsample_bilinear.cpp                                             */
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
 * @file test_upsample_bilinear.cpp
 * @brief Tests the reference bilinear upsample against hand values and its backward as its adjoint.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/upsample_bilinear.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns the span of an upsample of a dense NHWC input by scale_h x scale_w. */
        static Spatial::WindowSpan upsample_span(int64_t batch, int64_t height, int64_t width,
                                                 int64_t channels, int64_t scale_h,
                                                 int64_t scale_w)
        {
            Spatial::WindowSpan span = nhwc_span(batch, height, width, channels,
                                                 Spatial::ConvGeometry{}, 0);
            span.out_h = height * scale_h;
            span.out_w = width * scale_w;
            span.out_row_stride = span.out_w * channels;
            span.out_batch_stride = span.out_h * span.out_w * channels;
            return span;
        }

        TEST(Small_ReferenceUpsampleBilinear, InterpolatesAlongTheColumns)
        {
            const Spatial::WindowSpan span = upsample_span(1, 1, 2, 1, 1, 2);
            const std::vector<double> in{1.0, 5.0};
            std::vector<double> out(4, 0.0);

            Kernels::Reference::upsample_bilinear(span, 1, 2, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{1, 2, 4, 5}));
        }

        TEST(Small_ReferenceUpsampleBilinear, InterpolatesAlongTheRows)
        {
            const Spatial::WindowSpan span = upsample_span(1, 2, 1, 1, 2, 1);
            const std::vector<double> in{1.0, 5.0};
            std::vector<double> out(4, 0.0);

            Kernels::Reference::upsample_bilinear(span, 2, 1, in.data(), out.data());

            EXPECT_EQ(out, (std::vector<double>{1, 2, 4, 5}));
        }

        TEST(Small_ReferenceUpsampleBilinear, AScaleOfOneCopiesTheInput)
        {
            const Spatial::WindowSpan span = upsample_span(2, 3, 2, 2, 1, 1);
            const std::vector<double> in = small_integers<double>(span.input_elements(), 1);
            std::vector<double> out(in.size(), 0.0);

            Kernels::Reference::upsample_bilinear(span, 1, 1, in.data(), out.data());

            EXPECT_EQ(out, in);
        }

        TEST(Small_ReferenceUpsampleBilinear, TheBackwardIsTheAdjoint)
        {
            const Spatial::WindowSpan span = upsample_span(2, 3, 4, 2, 2, 4);
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            std::vector<double> y(static_cast<std::size_t>(span.output_elements()));
            std::vector<double> dx(static_cast<std::size_t>(span.input_elements()));

            Kernels::Reference::upsample_bilinear(span, 2, 4, x.data(), y.data());
            Kernels::Reference::upsample_bilinear_backward(span, 2, 4, dy.data(), dx.data());

            const double lhs = std::inner_product(y.begin(), y.end(), dy.begin(), 0.0);
            const double rhs = std::inner_product(x.begin(), x.end(), dx.begin(), 0.0);
            // The weights are multiples of 1 / 32 and the factors small integers, so both sums are exact.
            EXPECT_EQ(lhs, rhs);
        }
    } // namespace Test
} // namespace SushiBLAS
