/**************************************************************************/
/* test_orthonormal_sweep.cpp                                             */
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
 * @file test_orthonormal_sweep.cpp
 * @brief Tests the projection coefficient and the normalising scale of the orthonormalisation sweep.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds a dot product, a squared norm and the coefficient they give. */
            struct CoefficientRow
            {
                double dot;
                double norm_sq;
                double expected;
            };

            /** @brief Holds a gain, a squared norm, its root and the scale they give. */
            struct ScaleRow
            {
                double gain;
                double norm_sq;
                double root;
                double expected;
            };
        } // namespace

        class Small_OrthonormalProjection : public ::testing::TestWithParam<CoefficientRow>
        {
        };

        TEST_P(Small_OrthonormalProjection, TheCoefficientIsTheDotOverTheSquaredNormOrZero)
        {
            const CoefficientRow row = GetParam();
            EXPECT_EQ(Kernels::projection_coefficient(row.dot, row.norm_sq), row.expected);
        }

        // 6 / 4 = 1.5; a negative dot gives -1.5; a squared norm at or below the tiny bound counts as zero.
        INSTANTIATE_TEST_SUITE_P(, Small_OrthonormalProjection,
                                 ::testing::Values(CoefficientRow{6.0, 4.0, 1.5},
                                                   CoefficientRow{-6.0, 4.0, -1.5},
                                                   CoefficientRow{0.0, 4.0, 0.0},
                                                   CoefficientRow{6.0, 0.0, 0.0},
                                                   CoefficientRow{6.0, Kernels::tiny_norm_sq<double>(), 0.0}));

        class Small_OrthonormalScale : public ::testing::TestWithParam<ScaleRow>
        {
        };

        TEST_P(Small_OrthonormalScale, TheScaleIsTheGainOverTheNormOrTheGain)
        {
            const ScaleRow row = GetParam();
            EXPECT_EQ(Kernels::normalising_scale(row.gain, row.norm_sq, row.root), row.expected);
        }

        // Norm 2 (squared 4) and gain 3 give 1.5; a zero vector keeps the gain and never divides.
        INSTANTIATE_TEST_SUITE_P(, Small_OrthonormalScale,
                                 ::testing::Values(ScaleRow{3.0, 4.0, 2.0, 1.5},
                                                   ScaleRow{1.0, 4.0, 2.0, 0.5},
                                                   ScaleRow{3.0, 0.0, 0.0, 3.0},
                                                   ScaleRow{3.0, Kernels::tiny_norm_sq<double>(), 1e-12, 3.0}));
    } // namespace Test
} // namespace SushiBLAS
