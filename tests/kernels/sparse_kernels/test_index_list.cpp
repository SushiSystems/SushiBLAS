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
            /** @brief Returns what an IndexList of the zero-based positions, stored in Index from base, reads at entry. */
            template<typename Index>
            int64_t position_read(int64_t base, int64_t entry, const std::vector<int64_t>& zero_based)
            {
                std::vector<Index> stored;
                for (const int64_t position : zero_based)
                    stored.push_back(static_cast<Index>(position + base));
                const Kernels::IndexList<Index> list{static_cast<int64_t>(stored.size()), 8, base,
                                                     stored.data()};
                return list.at(entry);
            }

            /** @brief One stored list and the dense positions it must read. */
            struct PositionRow
            {
                    const char* name;                 /**< The row suffix. */
                    int64_t base;                     /**< The first index the list uses. */
                    std::vector<int64_t> positions;   /**< The zero-based positions the list holds. */
                    int64_t entry;                    /**< The entry read. */
                    int64_t (*read)(int64_t, int64_t, const std::vector<int64_t>&); /**< Reads the list stored in the row type. */
            };

            /**
             * @brief Returns every entry of the positions 4, 0 and 7 under base zero and one in
             *        INT64, and every entry of the positions 6, 1 and 4 (stored 7, 2, 5 one-based)
             *        in INT32 and INT64.
             */
            std::vector<PositionRow> position_rows()
            {
                const std::vector<int64_t> spread{4, 0, 7};
                const std::vector<int64_t> stored_7_2_5{6, 1, 4};
                return {{"ZeroBasedFirst", 0, spread, 0, &position_read<int64_t>},
                        {"ZeroBasedSecond", 0, spread, 1, &position_read<int64_t>},
                        {"ZeroBasedThird", 0, spread, 2, &position_read<int64_t>},
                        {"OneBasedFirst", 1, spread, 0, &position_read<int64_t>},
                        {"OneBasedSecond", 1, spread, 1, &position_read<int64_t>},
                        {"OneBasedThird", 1, spread, 2, &position_read<int64_t>},
                        {"OneBasedInt32Stored7First", 1, stored_7_2_5, 0, &position_read<int32_t>},
                        {"OneBasedInt32Stored2Second", 1, stored_7_2_5, 1, &position_read<int32_t>},
                        {"OneBasedInt32Stored5Third", 1, stored_7_2_5, 2, &position_read<int32_t>},
                        {"OneBasedInt64Stored7First", 1, stored_7_2_5, 0, &position_read<int64_t>},
                        {"OneBasedInt64Stored2Second", 1, stored_7_2_5, 1, &position_read<int64_t>},
                        {"OneBasedInt64Stored5Third", 1, stored_7_2_5, 2, &position_read<int64_t>}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
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

            EXPECT_EQ(row.read(row.base, row.entry, row.positions), row.positions[row.entry]);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_IndexList, ::testing::ValuesIn(position_rows()),
                                 position_row_name);
    } // namespace Test
} // namespace SushiBLAS
