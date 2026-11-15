/**************************************************************************/
/* test_rows.cpp                                                          */
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
 * @file test_rows.cpp
 * @brief Tests the reference take_rows and add_rows against hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/index_kernels/reference/rows.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds a 3 x 2 table, row-major: rows (1, 2), (3, 4) and (5, 6). */
        static const std::vector<double>& table_values()
        {
            static const std::vector<double> values{1, 2, 3, 4, 5, 6};
            return values;
        }

        TEST(Small_ReferenceRows, TakeRowsCopiesTheRowsTheIndexNamesInItsOrder)
        {
            const std::vector<int64_t> index{2, 0, 2};
            std::vector<double> rows(6, -1.0);

            Kernels::Reference::take_rows<double, int64_t>(
                Kernels::MatrixSpan{3, 2, 2, 1}, table_values().data(), index.data(), 1,
                Kernels::MatrixSpan{3, 2, 2, 1}, rows.data());

            EXPECT_EQ(rows, (std::vector<double>{5, 6, 1, 2, 5, 6}));
        }

        TEST(Small_ReferenceRows, TakeRowsWritesZerosForAnIndexOutsideTheTable)
        {
            const std::vector<int32_t> index{-1, 3, 1};
            std::vector<double> rows(6, -1.0);

            Kernels::Reference::take_rows<double, int32_t>(
                Kernels::MatrixSpan{3, 2, 2, 1}, table_values().data(), index.data(), 1,
                Kernels::MatrixSpan{3, 2, 2, 1}, rows.data());

            EXPECT_EQ(rows, (std::vector<double>{0, 0, 0, 0, 3, 4}));
        }

        TEST(Small_ReferenceRows, AddRowsSumsTheRowsOfARepeatedIndexAndOverwritesTheTable)
        {
            const std::vector<double> rows{1, 10, 2, 20, 4, 40};
            const std::vector<int64_t> index{1, 1, 0};
            std::vector<double> table(6, 99.0);

            Kernels::Reference::add_rows<double, int64_t>(
                Kernels::MatrixSpan{3, 2, 2, 1}, rows.data(), index.data(), 1,
                Kernels::MatrixSpan{3, 2, 2, 1}, table.data(), 0.0);

            EXPECT_EQ(table, (std::vector<double>{4, 40, 3, 30, 0, 0}));
        }

        TEST(Small_ReferenceRows, AddRowsScalesTheTableByBetaBeforeAdding)
        {
            const std::vector<double> rows{1, 10, 100, 1000};
            const std::vector<int64_t> index{2, 7};
            std::vector<double> table{2, 4, 6, 8, 10, 12};

            Kernels::Reference::add_rows<double, int64_t>(
                Kernels::MatrixSpan{2, 2, 2, 1}, rows.data(), index.data(), 1,
                Kernels::MatrixSpan{3, 2, 2, 1}, table.data(), 0.5);

            // The index 7 selects no row of the table, so the second row of rows is skipped.
            EXPECT_EQ(table, (std::vector<double>{1, 2, 3, 4, 6, 16}));
        }
    } // namespace Test
} // namespace SushiBLAS
