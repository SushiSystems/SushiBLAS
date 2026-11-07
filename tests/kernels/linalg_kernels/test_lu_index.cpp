/**************************************************************************/
/* test_lu_index.cpp                                                      */
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
 * @file test_lu_index.cpp
 * @brief Tests the pivot rule, the singular bound and the cell coordinates of the batched LU, on the host.
 * @author Mustafa Garip
 */

#include <limits>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/linalg_kernels/lu_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds a step, a trailing width, a flat index and the cell it names. */
            struct TrailingRow
            {
                std::size_t step;
                std::size_t width;
                std::size_t flat;
                std::size_t row;
                std::size_t column;
            };

            /** @brief Holds a flat solve index, a column count and the matrix and column it names. */
            struct SolveRow
            {
                std::size_t flat;
                std::size_t rhs_count;
                std::size_t matrix;
                std::size_t column;
            };
        } // namespace

        TEST(Small_LuPivotSearch, ATieOfMagnitudesKeepsTheLowestRow)
        {
            // Column 0 of a 4 * 4 matrix is 1, -5, 5, 2: rows 1 and 2 tie at magnitude 5.
            const double a[16] = {1, 0, 0, 0, -5, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0};
            const Indexing::LuPivot<double> best = Indexing::lu_pivot_search(a, 4, 0);
            EXPECT_EQ(best.row, 1U);
            EXPECT_EQ(best.magnitude, 5.0);
        }

        TEST(Small_LuPivotSearch, TheDiagonalWinsWhenNothingBelowIsLarger)
        {
            // Column 1 holds 9 on the diagonal and 3 below it.
            const double a[9] = {0, 0, 0, 0, 9, 0, 0, 3, 0};
            const Indexing::LuPivot<double> best = Indexing::lu_pivot_search(a, 3, 1);
            EXPECT_EQ(best.row, 1U);
            EXPECT_EQ(best.magnitude, 9.0);
        }

        TEST(Small_LuPivotSearch, ARowAboveTheStepIsNeverSearched)
        {
            // Column 1 holds 100 in row 0, above step 1, and 3 in row 1 and 4 in row 2.
            const double a[9] = {0, 100, 0, 0, 3, 0, 0, 4, 0};
            const Indexing::LuPivot<double> best = Indexing::lu_pivot_search(a, 3, 1);
            EXPECT_EQ(best.row, 2U);
            EXPECT_EQ(best.magnitude, 4.0);
        }

        TEST(Small_LuPivotSearch, ANegativeDiagonalIsComparedByMagnitude)
        {
            const double a[4] = {-7, 0, 6, 0};
            const Indexing::LuPivot<double> best = Indexing::lu_pivot_search(a, 2, 0);
            EXPECT_EQ(best.row, 0U);
            EXPECT_EQ(best.magnitude, 7.0);
        }

        TEST(Small_LuPivotSearch, ANotANumberBelowTheDiagonalIsNeverChosen)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double a[4] = {2, 0, nan, 0};
            const Indexing::LuPivot<double> best = Indexing::lu_pivot_search(a, 2, 0);
            EXPECT_EQ(best.row, 0U);
        }

        TEST(Small_LuThreshold, TheBoundIsNTimesEpsilonTimesTheLargestElement)
        {
            EXPECT_EQ(Indexing::lu_threshold<double>(4, 8.0),
                      4.0 * std::numeric_limits<double>::epsilon() * 8.0);
            EXPECT_EQ(Indexing::lu_threshold<float>(0, 8.0f), 0.0f);
        }

        TEST(Small_LuThreshold, APivotMustBeStrictlyAboveTheBound)
        {
            EXPECT_TRUE(Indexing::lu_pivot_usable(2.0, 1.0));
            EXPECT_FALSE(Indexing::lu_pivot_usable(1.0, 1.0));
            EXPECT_FALSE(Indexing::lu_pivot_usable(0.5, 1.0));
            EXPECT_FALSE(Indexing::lu_pivot_usable(std::numeric_limits<double>::quiet_NaN(), 1.0));
        }

        class Small_LuTrailingCell : public ::testing::TestWithParam<TrailingRow>
        {
        };

        TEST_P(Small_LuTrailingCell, TheFlatIndexWalksTheBlockBelowAndRightOfTheStepRowByRow)
        {
            const TrailingRow row = GetParam();
            const Indexing::LuCell cell = Indexing::lu_trailing_cell(row.step, row.width, row.flat);
            EXPECT_EQ(cell.row, row.row);
            EXPECT_EQ(cell.column, row.column);
        }

        // Step 1 of a 4 * 4 matrix leaves the 2 * 2 block at rows and columns 2 and 3.
        INSTANTIATE_TEST_SUITE_P(, Small_LuTrailingCell,
                                 ::testing::Values(TrailingRow{1, 2, 0, 2, 2},
                                                   TrailingRow{1, 2, 1, 2, 3},
                                                   TrailingRow{1, 2, 2, 3, 2},
                                                   TrailingRow{1, 2, 3, 3, 3},
                                                   TrailingRow{0, 1, 0, 1, 1}));

        class Small_LuSolveCell : public ::testing::TestWithParam<SolveRow>
        {
        };

        TEST_P(Small_LuSolveCell, TheFlatIndexWalksTheColumnsOfOneMatrixThenTheNext)
        {
            const SolveRow row = GetParam();
            const Indexing::LuSolveCell cell = Indexing::lu_solve_cell(row.flat, row.rhs_count);
            EXPECT_EQ(cell.matrix, row.matrix);
            EXPECT_EQ(cell.column, row.column);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_LuSolveCell,
                                 ::testing::Values(SolveRow{0, 3, 0, 0},
                                                   SolveRow{2, 3, 0, 2},
                                                   SolveRow{3, 3, 1, 0},
                                                   SolveRow{7, 3, 2, 1},
                                                   SolveRow{5, 1, 5, 0}));
    } // namespace Test
} // namespace SushiBLAS
