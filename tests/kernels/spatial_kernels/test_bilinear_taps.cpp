/**************************************************************************/
/* test_bilinear_taps.cpp                                                 */
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
 * @file test_bilinear_taps.cpp
 * @brief Tests the bilinear taps, the blend and the output ranges against hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/spatial_kernels/bilinear_taps.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Bilinear = Kernels::Bilinear;

        /** @brief Holds one tap call and the tap it must return. */
        struct TapRow
        {
            int64_t out_index;
            int64_t scale;
            int64_t extent;
            int64_t low;
            int64_t high;
            double weight;
        };

        class Small_BilinearTap : public ::testing::TestWithParam<TapRow>
        {
        };

        TEST_P(Small_BilinearTap, TapPlacesTheOutputCentreBetweenTwoSourcePixels)
        {
            const TapRow row = GetParam();

            const Bilinear::Tap<double> tap =
                Bilinear::tap<double>(row.out_index, row.scale, row.extent);

            EXPECT_EQ(tap.low, row.low);
            EXPECT_EQ(tap.high, row.high);
            // The weight is a multiple of 1 / (2 * scale), a power of two here, so it is exact.
            EXPECT_EQ(tap.weight, row.weight);
        }

        // Scale 2 over 2 pixels: centres at -0.25, 0.25, 0.75, 1.25; scale 1 copies at weight 0.
        INSTANTIATE_TEST_SUITE_P(, Small_BilinearTap,
                                 ::testing::Values(TapRow{0, 2, 2, 0, 0, 0.75},
                                                   TapRow{1, 2, 2, 0, 1, 0.25},
                                                   TapRow{2, 2, 2, 0, 1, 0.75},
                                                   TapRow{3, 2, 2, 1, 1, 0.25},
                                                   TapRow{0, 1, 3, 0, 1, 0.0},
                                                   TapRow{2, 1, 3, 2, 2, 0.0}));

        TEST(Small_BilinearBlend, ABlendIsTheWeightedSumOfItsTwoValues)
        {
            EXPECT_EQ(Bilinear::blend(2.0, 4.0, 0.25), 2.5);
        }

        TEST(Small_BilinearBlend, AZeroWeightReturnsTheFirstValueWhateverTheSecondIs)
        {
            const double infinity = std::numeric_limits<double>::infinity();

            EXPECT_EQ(Bilinear::blend(2.0, infinity, 0.0), 2.0);
        }

        TEST(Small_BilinearAxisWeight, AnIndexThatBothSourcesShareReceivesTheWholeWeight)
        {
            const Bilinear::Tap<double> clamped = Bilinear::tap<double>(0, 2, 2);

            EXPECT_EQ(Bilinear::axis_weight(clamped, 0), 1.0);
        }

        TEST(Small_BilinearAxisWeight, TheTwoSourcesOfATapReceiveTheTwoShares)
        {
            const Bilinear::Tap<double> tap = Bilinear::tap<double>(1, 2, 2);

            EXPECT_EQ(Bilinear::axis_weight(tap, 0), 0.75);
            EXPECT_EQ(Bilinear::axis_weight(tap, 1), 0.25);
        }

        TEST(Small_BilinearAxisWeight, ASourceOutsideTheTapReceivesNothing)
        {
            const Bilinear::Tap<double> tap = Bilinear::tap<double>(1, 2, 3);

            EXPECT_EQ(Bilinear::axis_weight(tap, 2), 0.0);
        }

        TEST(Small_BilinearRange, TheFirstOutputOfAnIndexIsOneSourceWidthBelowItsOwn)
        {
            EXPECT_EQ(Bilinear::first_output(0, 2), 0);
            EXPECT_EQ(Bilinear::first_output(1, 2), 0);
            EXPECT_EQ(Bilinear::first_output(3, 2), 4);
        }

        TEST(Small_BilinearRange, TheEndOutputOfAnIndexIsClippedToTheOutputExtent)
        {
            EXPECT_EQ(Bilinear::end_output(1, 2, 8), 6);
            EXPECT_EQ(Bilinear::end_output(3, 2, 8), 8);
        }
    } // namespace Test
} // namespace SushiBLAS
