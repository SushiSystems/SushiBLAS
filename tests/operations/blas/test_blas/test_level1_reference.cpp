/**************************************************************************/
/* test_level1_reference.cpp                                              */
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
 * @file test_level1_reference.cpp
 * @brief Tests the BLAS Level 1 reference kernels against hand-computed results.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/reference/level1/level1.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Tests that the reference rot turns each pair by the cosine 0.6 and the sine 0.8,
         *        a rotation no other test uses.
         *
         * The expected pairs are worked out by hand; each is within three FLOAT32 roundings of
         * |c| * |x| + |s| * |y|, the terms that form it.
         */
        TEST(Small_Level1Reference, RotAppliesThePlaneRotation)
        {
            constexpr double K_COSINE = 0.6;
            constexpr double K_SINE = 0.8;
            constexpr std::size_t K_ROUNDINGS = 3;

            std::vector<float> x = {1.0f, 2.0f};
            std::vector<float> y = {3.0f, 4.0f};
            const std::vector<double> want_x = {3.0, 4.4};
            const std::vector<double> want_y = {1.0, 0.8};

            Kernels::Reference::rot<float>(2, x.data(), 1, y.data(), 1, static_cast<float>(K_COSINE),
                                           static_cast<float>(K_SINE));

            for (std::size_t i = 0; i < x.size(); ++i)
            {
                const double magnitude = K_COSINE * std::abs(want_x[i]) + K_SINE * std::abs(want_y[i]);
                const double bound = static_cast<double>(K_ROUNDINGS) *
                                     TestSupport::unit_roundoff<float>() * (magnitude + 4.0);

                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(x[i], want_x[i], bound)))
                    << "x[" << i << "]";
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(y[i], want_y[i], bound)))
                    << "y[" << i << "]";
            }
        }
    } // namespace Test
} // namespace SushiBLAS
