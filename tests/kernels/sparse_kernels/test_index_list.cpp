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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/index_list.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief One stored list and the dense positions it must read. */
            struct PositionRow
            {
                    const char* name;                 /**< The row's suffix. */
                    int64_t base;                     /**< The first index the list uses. */
                    int64_t entry;                    /**< The entry read. */
                    int64_t expected;                 /**< The dense position of that entry. */
            };

            /** @brief Returns every entry of the positions 4, 0 and 7 under base zero and one. */
            std::vector<PositionRow> position_rows()
            {
                return {{"ZeroBasedFirst", 0, 0, 4}, {"ZeroBasedSecond", 0, 1, 0},
                        {"ZeroBasedThird", 0, 2, 7}, {"OneBasedFirst", 1, 0, 4},
                        {"OneBasedSecond", 1, 1, 0},  {"OneBasedThird", 1, 2, 7}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string position_row_name(const ::testing::TestParamInfo<PositionRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Reads one entry of a list at a time. */
        class Small_IndexList : public ::testing::TestWithParam<PositionRow>
        {

        };

        TEST_P(Small_IndexList, ReadsEachPositionLessTheBase)
        {
            const PositionRow& row = GetParam();
            const std::vector<int64_t> stored{4 + row.base, 0 + row.base, 7 + row.base};
            const Kernels::IndexList<int64_t> list{3, 8, row.base, stored.data()};

            EXPECT_EQ(list.at(row.entry), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_IndexList, ::testing::ValuesIn(position_rows()),
                                 position_row_name);
    } // namespace Test
} // namespace SushiBLAS
