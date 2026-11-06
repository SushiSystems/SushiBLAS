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

            /** @brief Returns what a 3 x 4 pattern stored in Index from base reads: begin and end of row 0, row 1 and the end of row 2, and columns 1 and 2. */
            template<typename Index>
            std::vector<int64_t> read_by_the_pattern(int64_t base)
            {
                const std::vector<Index> offsets = stored_at<Index>({0, 2, 2, 3}, static_cast<Index>(base));
                const std::vector<Index> columns = stored_at<Index>({1, 3, 0}, static_cast<Index>(base));
                const Kernels::CsrPattern<Index> pattern{3, 4, 3, base, offsets.data(), columns.data()};

                return {pattern.row_begin(0), pattern.row_end(0), pattern.row_begin(1),
                        pattern.row_end(1),   pattern.row_end(2), pattern.column(1),
                        pattern.column(2)};
            }

            /** @brief One index base and type of the pattern. */
            struct PatternRow
            {
                    const char* name;         /**< The row's suffix. */
                    int64_t base;             /**< The first index the arrays use. */
                    std::vector<int64_t> (*read)(int64_t); /**< Returns what the pattern stored in the row's type reads. */
            };

            /** @brief Returns the four combinations of base and index type. */
            std::vector<PatternRow> pattern_rows()
            {
                return {{"ZeroBasedInt32", 0, &read_by_the_pattern<int32_t>},
                        {"ZeroBasedInt64", 0, &read_by_the_pattern<int64_t>},
                        {"OneBasedInt32", 1, &read_by_the_pattern<int32_t>},
                        {"OneBasedInt64", 1, &read_by_the_pattern<int64_t>}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string pattern_row_name(const ::testing::TestParamInfo<PatternRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the pattern check one base and index type at a time. */
        class Small_CsrPattern : public ::testing::TestWithParam<PatternRow>
        {

        };

        TEST_P(Small_CsrPattern, ReadsItsArraysLessTheBase)
        {
            // Row 0 spans entries 0 to 2, row 1 is empty at 2, row 2 ends at 3; columns 1 and 2 are 3 and 0.
            EXPECT_EQ(GetParam().read(GetParam().base), (std::vector<int64_t>{0, 2, 2, 2, 3, 3, 0}));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_CsrPattern, ::testing::ValuesIn(pattern_rows()),
                                 pattern_row_name);
    } // namespace Test
} // namespace SushiBLAS
