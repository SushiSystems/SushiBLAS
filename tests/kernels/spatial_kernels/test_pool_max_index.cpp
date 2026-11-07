/**************************************************************************/
/* test_pool_max_index.cpp                                                */
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
 * @file test_pool_max_index.cpp
 * @brief Tests the max-pool step that decides whether a tap replaces the running maximum.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiBLAS/spatial_kernels/pool_max_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds one takes_candidate call over float. */
        struct CandidateRow
        {
            int64_t best_index;
            float best;
            float candidate;
            bool expected;
        };

        class Small_PoolMaxIndexRows : public ::testing::TestWithParam<CandidateRow>
        {
        };

        TEST_P(Small_PoolMaxIndexRows, TakesCandidateKeepsTheFirstLiveTapTheLargerValueAndTheNaN)
        {
            const CandidateRow row = GetParam();

            EXPECT_EQ(Indexing::takes_candidate(row.best_index, row.best, row.candidate),
                      row.expected);
        }

        constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();
        constexpr float K_LOWEST = std::numeric_limits<float>::lowest();

        // Index -1 takes any tap; a held number yields to a larger or NaN tap; a held NaN stays.
        INSTANTIATE_TEST_SUITE_P(, Small_PoolMaxIndexRows,
                                 ::testing::Values(CandidateRow{-1, K_LOWEST, K_LOWEST, true},
                                                   CandidateRow{-1, K_LOWEST, K_NAN, true},
                                                   CandidateRow{0, 1.0f, 2.0f, true},
                                                   CandidateRow{0, 1.0f, 1.0f, false},
                                                   CandidateRow{0, 1.0f, 0.0f, false},
                                                   CandidateRow{0, 1.0f, K_NAN, true},
                                                   CandidateRow{3, K_NAN, 5.0f, false},
                                                   CandidateRow{3, K_NAN, K_NAN, false},
                                                   CandidateRow{2, K_LOWEST, K_LOWEST, false}));

        TEST(Small_PoolMaxIndex, TakesCandidateWorksOverDoubleToo)
        {
            EXPECT_TRUE(Indexing::takes_candidate<double>(0, 1.0, 2.0));
            EXPECT_FALSE(Indexing::takes_candidate<double>(0, 2.0, 1.0));
        }
    }
}
