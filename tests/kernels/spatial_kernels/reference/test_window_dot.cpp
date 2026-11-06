/**************************************************************************/
/* test_window_dot.cpp                                                    */
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
 * @file test_window_dot.cpp
 * @brief Tests the reference window product against hand values and its backward as its adjoint.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// Relative bound on the two sides of an adjoint identity, which add the same products in different orders.
        constexpr double K_ADJOINT_TOLERANCE = 1e-12;

        TEST(Small_ReferenceWindowDot, WeighsEachTapOfOneChannel)
        {
            Spatial::ConvGeometry geometry;
            geometry.kernel_h = 1;
            geometry.kernel_w = 2;
            const Spatial::WindowSpan span = nhwc_span(1, 1, 3, 1, geometry, 0);
            const std::vector<float> in{1.0f, 2.0f, 3.0f};
            const std::vector<float> weights{10.0f, 100.0f};
            std::vector<float> out(2, 0.0f);

            Kernels::Reference::window_dot(span, Spatial::TapSpan{2, 1, 1}, in.data(),
                                           weights.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{210.0f, 320.0f}));
        }

        TEST(Small_ReferenceWindowDot, TheBackwardIsTheAdjoint)
        {
            const Spatial::WindowSpan span = nhwc_span(2, 7, 6, 3, uneven_geometry(), 1);
            const Spatial::TapSpan taps{span.geometry.kernel_w * 3, 3, 1};
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> w = small_integers<double>(span.patch_elements(), 2);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 3);
            std::vector<double> y(static_cast<std::size_t>(span.output_elements()));
            std::vector<double> dx(static_cast<std::size_t>(span.input_elements()));

            Kernels::Reference::window_dot(span, taps, x.data(), w.data(), y.data());
            Kernels::Reference::window_dot_backward(span, taps, dy.data(), w.data(), dx.data());

            const double lhs = std::inner_product(y.begin(), y.end(), dy.begin(), 0.0);
            const double rhs = std::inner_product(x.begin(), x.end(), dx.begin(), 0.0);
            EXPECT_NEAR(lhs, rhs, K_ADJOINT_TOLERANCE * std::max(1.0, std::abs(lhs)));
        }
    } // namespace Test
} // namespace SushiBLAS
