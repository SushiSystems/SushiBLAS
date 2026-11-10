/**************************************************************************/
/* test_triangle_index.cpp                                                */
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
 * @file test_triangle_index.cpp
 * @brief Tests the stored-triangle test and the substitution row order of the triangular sparse kernels.
 * @author Mustafa Garip
 */

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/triangle_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds one in_stored_triangle call. */
        struct StoredTriangleRow
        {
            bool lower;
            int64_t col;
            int64_t row;
            bool expected;
        };

        class Small_TriangleIndexMembership : public ::testing::TestWithParam<StoredTriangleRow>
        {
        };

        TEST_P(Small_TriangleIndexMembership, TheStoredTriangleExcludesTheDiagonal)
        {
            const StoredTriangleRow row = GetParam();

            EXPECT_EQ(Indexing::in_stored_triangle(row.lower, row.col, row.row), row.expected);
        }

        // Row 1: a lower triangle holds column 0, an upper one holds column 2; neither holds column 1.
        INSTANTIATE_TEST_SUITE_P(, Small_TriangleIndexMembership,
                                 ::testing::Values(StoredTriangleRow{true, 0, 1, true},
                                                   StoredTriangleRow{true, 1, 1, false},
                                                   StoredTriangleRow{true, 2, 1, false},
                                                   StoredTriangleRow{false, 0, 1, false},
                                                   StoredTriangleRow{false, 1, 1, false},
                                                   StoredTriangleRow{false, 2, 1, true}));

        /** @brief Holds a row count and the direction its rows are visited in. */
        struct RowOrderRow
        {
            int64_t n;
            bool ascending;
        };

        /** @brief Returns the rows 0 .. n - 1, or n - 1 .. 0, as a plain list. */
        static std::vector<int64_t> expected_order(const RowOrderRow& row)
        {
            std::vector<int64_t> rows(static_cast<std::size_t>(row.n));
            std::iota(rows.begin(), rows.end(), int64_t{0});
            if (!row.ascending)
                std::reverse(rows.begin(), rows.end());
            return rows;
        }

        /** @brief Returns the rows ordered_row visits over the steps 0 .. n - 1. */
        static std::vector<int64_t> visited_rows(const RowOrderRow& row)
        {
            std::vector<int64_t> rows;
            for (int64_t step = 0; step < row.n; ++step)
                rows.push_back(Indexing::ordered_row(step, row.n, row.ascending));
            return rows;
        }

        class Small_TriangleIndexOrder : public ::testing::TestWithParam<RowOrderRow>
        {
        };

        TEST_P(Small_TriangleIndexOrder, OrderedRowVisitsEveryRowOnceInTheGivenDirection)
        {
            EXPECT_EQ(visited_rows(GetParam()), expected_order(GetParam()));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_TriangleIndexOrder,
                                 ::testing::Values(RowOrderRow{0, true}, RowOrderRow{0, false},
                                                   RowOrderRow{1, true}, RowOrderRow{1, false},
                                                   RowOrderRow{2, true}, RowOrderRow{3, false},
                                                   RowOrderRow{5, true}, RowOrderRow{5, false},
                                                   RowOrderRow{9, true}, RowOrderRow{9, false}));
    } // namespace Test
} // namespace SushiBLAS
