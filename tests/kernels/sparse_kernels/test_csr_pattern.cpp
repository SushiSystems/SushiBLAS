/**************************************************************************/
/* test_csr_pattern.cpp                                                   */
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
 * @file test_csr_pattern.cpp
 * @brief Tests that a CsrPattern reads its rows and columns under either index base.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the values of a zero-based array moved up to start at base. */
            template<typename Index>
            std::vector<Index> stored_at(const std::vector<int64_t>& zero_based, Index base)
            {
                std::vector<Index> out;
                for (const int64_t value : zero_based)
                    out.push_back(static_cast<Index>(value) + base);
                return out;
            }

            /** @brief Holds a zero-based CSR structure and what a pattern over it must read. */
            struct PatternData
            {
                    int64_t cols;                      /**< The number of columns. */
                    std::vector<int64_t> offsets;      /**< The zero-based row offsets. */
                    std::vector<int64_t> columns;      /**< The zero-based column of each entry. */
                    std::vector<int64_t> expected;     /**< Begin and end of each row, then each column. */
            };

            /**
             * @brief Returns what a pattern over the data, stored in Index from base, reads: the
             *        begin and the end of each row, then the column of each entry.
             */
            template<typename Index>
            std::vector<int64_t> read_by_the_pattern(const PatternData& data, int64_t base)
            {
                const std::vector<Index> offsets = stored_at<Index>(data.offsets, static_cast<Index>(base));
                const std::vector<Index> columns = stored_at<Index>(data.columns, static_cast<Index>(base));
                const int64_t rows = static_cast<int64_t>(data.offsets.size()) - 1;
                const Kernels::CsrPattern<Index> pattern{
                    rows, data.cols, static_cast<int64_t>(data.columns.size()), base,
                    offsets.data(), columns.data()};

                std::vector<int64_t> read;
                for (int64_t row = 0; row < rows; ++row)
                {
                    read.push_back(pattern.row_begin(row));
                    read.push_back(pattern.row_end(row));
                }
                for (int64_t entry = 0; entry < static_cast<int64_t>(data.columns.size()); ++entry)
                    read.push_back(pattern.column(entry));
                return read;
            }

            /** @brief One structure, index base and index type of the pattern. */
            struct PatternRow
            {
                    const char* name;         /**< The row suffix. */
                    int64_t base;             /**< The first index the arrays use. */
                    PatternData data;         /**< The structure the arrays hold and what it reads. */
                    std::vector<int64_t> (*read)(const PatternData&, int64_t); /**< Reads the data stored in the row type. */
            };

            /**
             * @brief Returns the combinations of structure, base and index type.
             *
             * The first structure has 3 rows and 3 entries with row 1 empty; the second has 3 rows
             * and 5 entries with row 1 empty and the last row longest.
             */
            std::vector<PatternRow> pattern_rows()
            {
                // Row 0 spans entries 0 to 2, row 1 is empty at 2, row 2 ends at 3; columns 1, 3, 0.
                const PatternData small{4, {0, 2, 2, 3}, {1, 3, 0}, {0, 2, 2, 2, 2, 3, 1, 3, 0}};
                // Row 0 spans entries 0 to 2, row 1 is empty at 2, row 2 spans 2 to 5; columns 0, 3, 1, 2, 4.
                const PatternData wide{5, {0, 2, 2, 5}, {0, 3, 1, 2, 4}, {0, 2, 2, 2, 2, 5, 0, 3, 1, 2, 4}};
                return {{"SmallZeroBasedInt32", 0, small, &read_by_the_pattern<int32_t>},
                        {"SmallZeroBasedInt64", 0, small, &read_by_the_pattern<int64_t>},
                        {"SmallOneBasedInt32", 1, small, &read_by_the_pattern<int32_t>},
                        {"SmallOneBasedInt64", 1, small, &read_by_the_pattern<int64_t>},
                        {"WideZeroBasedInt32", 0, wide, &read_by_the_pattern<int32_t>},
                        {"WideZeroBasedInt64", 0, wide, &read_by_the_pattern<int64_t>},
                        {"WideOneBasedInt32", 1, wide, &read_by_the_pattern<int32_t>},
                        {"WideOneBasedInt64", 1, wide, &read_by_the_pattern<int64_t>}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
            std::string pattern_row_name(const ::testing::TestParamInfo<PatternRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the pattern check one structure, base and index type at a time. */
        class Small_CsrPattern : public ::testing::TestWithParam<PatternRow>
        {

        };

        TEST_P(Small_CsrPattern, ReadsItsArraysLessTheBase)
        {
            EXPECT_EQ(GetParam().read(GetParam().data, GetParam().base), GetParam().data.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_CsrPattern, ::testing::ValuesIn(pattern_rows()),
                                 pattern_row_name);
    } // namespace Test
} // namespace SushiBLAS
