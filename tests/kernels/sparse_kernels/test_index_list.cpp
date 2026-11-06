/**************************************************************************/
/* test_index_list.cpp                                                    */
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
 * @file test_index_list.cpp
 * @brief Tests that an IndexList reads the dense positions of its entries under either base.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/index_list.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_IndexList, ReadsEachPositionLessTheBase)
        {
            const std::vector<int32_t> zero_based{4, 0, 7};
            const std::vector<int64_t> one_based{5, 1, 8};
            const Kernels::IndexList<int32_t> zero{3, 8, 0, zero_based.data()};
            const Kernels::IndexList<int64_t> one{3, 8, 1, one_based.data()};

            for (int64_t entry = 0; entry < 3; ++entry)
                EXPECT_EQ(zero.at(entry), one.at(entry));
            EXPECT_EQ(zero.at(2), int64_t{7});
            EXPECT_EQ(one.extent, int64_t{8});
        }
    } // namespace Test
} // namespace SushiBLAS
