/**************************************************************************/
/* test_rotm_index.cpp                                                    */
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
 * @file test_rotm_index.cpp
 * @brief Tests the ROTM flag test and pair transformation against hand-derived values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/rotm_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /// @brief The first component of the pair the rows rotate.
            constexpr double K_W = 3.0;

            /// @brief The second component of the pair the rows rotate.
            constexpr double K_Z = 5.0;

            /// @brief The H entry every row multiplies w by into x; every product is exact in double.
            constexpr double K_H11 = 2.0;

            /// @brief The H entry every row multiplies w by into y.
            constexpr double K_H21 = 7.0;

            /// @brief The H entry every row multiplies z by into x.
            constexpr double K_H12 = 11.0;

            /// @brief The H entry every row multiplies z by into y.
            constexpr double K_H22 = 13.0;

            /** @brief Holds one flag and the pair it must give. */
            struct RotmRow
            {
                double flag;
                double expected_x;
                double expected_y;
            };

            /** @brief Checks that rotm_apply gives the expected pair for the row's flag. */
            void expect_pair(const RotmRow& row)
            {
                const Indexing::RotmPair<double> got = Indexing::rotm_apply(
                    row.flag, K_H11, K_H21, K_H12, K_H22, K_W, K_Z);
                EXPECT_EQ(got.x, row.expected_x);
                EXPECT_EQ(got.y, row.expected_y);
            }
        } // namespace

        class Small_RotmIndexApply : public ::testing::TestWithParam<RotmRow>
        {
        };

        TEST_P(Small_RotmIndexApply, TheFlagSelectsTheMatrixItApplies)
        {
            expect_pair(GetParam());
        }

        // Flag -1: (3*2 + 5*11, 3*7 + 5*13); flag 0: (3 + 5*11, 3*7 + 5); flag 1: (3*2 + 5, -3 + 5*13).
        INSTANTIATE_TEST_SUITE_P(, Small_RotmIndexApply,
                                 ::testing::Values(RotmRow{-1.0, 61.0, 86.0},
                                                   RotmRow{0.0, 58.0, 26.0},
                                                   RotmRow{1.0, 11.0, 62.0}));

        /** @brief Holds one flag and whether it selects the identity. */
        struct IdentityRow
        {
            double flag;
            bool identity;
        };

        class Small_RotmIndexIdentity : public ::testing::TestWithParam<IdentityRow>
        {
        };

        TEST_P(Small_RotmIndexIdentity, OnlyTheMinusTwoFlagIsTheIdentity)
        {
            EXPECT_EQ(Indexing::rotm_is_identity(GetParam().flag), GetParam().identity);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RotmIndexIdentity,
                                 ::testing::Values(IdentityRow{-2.0, true}, IdentityRow{-1.0, false},
                                                   IdentityRow{0.0, false}, IdentityRow{1.0, false},
                                                   IdentityRow{2.0, false}));
    } // namespace Test
} // namespace SushiBLAS
