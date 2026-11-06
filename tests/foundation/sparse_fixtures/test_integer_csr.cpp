/**************************************************************************/
/* test_integer_csr.cpp                                                   */
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
 * @file test_integer_csr.cpp
 * @brief Tests that integer_csr builds a valid CSR matrix of small integers and its dense twin.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns the index of a vector element as the unsigned type a vector takes.
        [[nodiscard]] static std::size_t at(int64_t index) noexcept
        {
            return static_cast<std::size_t>(index);
        }

        /// @brief Expects offsets that start at 0, end at nnz and never fall, and ascending columns inside the shape.
        static void expect_valid_csr(const IntegerCsr& m)
        {
            ASSERT_EQ(m.offsets.size(), at(m.rows) + 1);
            EXPECT_EQ(m.offsets.front(), int64_t{0});
            EXPECT_EQ(m.offsets.back(), m.nnz());
            EXPECT_EQ(m.columns.size(), m.values.size());
            for (int64_t i = 0; i < m.rows; ++i)
            {
                EXPECT_LE(m.offsets[at(i)], m.offsets[at(i + 1)]) << "row " << i;
                for (int64_t e = m.offsets[at(i)]; e < m.offsets[at(i + 1)]; ++e)
                {
                    EXPECT_GE(m.columns[at(e)], int64_t{0}) << "entry " << e;
                    EXPECT_LT(m.columns[at(e)], m.cols) << "entry " << e;
                    if (e > m.offsets[at(i)])
                    {
                        EXPECT_LT(m.columns[at(e - 1)], m.columns[at(e)]) << "entry " << e;
                    }
                }
            }
        }

        /// @brief Expects every diagonal position of a square matrix to be stored, with the value 1.
        static void expect_stored_diagonal_of_ones(const IntegerCsr& m)
        {
            for (int64_t i = 0; i < m.rows; ++i)
            {
                bool found = false;
                for (int64_t e = m.offsets[at(i)]; e < m.offsets[at(i + 1)]; ++e)
                {
                    if (m.columns[at(e)] == i)
                    {
                        found = true;
                        EXPECT_EQ(m.values[at(e)], 1.0) << "row " << i;
                    }
                }
                EXPECT_TRUE(found) << "row " << i << " stores no diagonal entry";
            }
        }

        /// @brief Expects dense() to hold, at the position of each stored entry, that entry's value.
        static void expect_dense_holds_each_stored_value(const IntegerCsr& m)
        {
            const std::vector<double> dense = m.dense();

            ASSERT_EQ(dense.size(), at(m.rows * m.cols));
            for (int64_t i = 0; i < m.rows; ++i)
            {
                for (int64_t e = m.offsets[at(i)]; e < m.offsets[at(i + 1)]; ++e)
                {
                    EXPECT_EQ(dense[at(i * m.cols + m.columns[at(e)])], m.values[at(e)])
                        << "entry " << e;
                }
            }
        }

        TEST(Small_IntegerCsr, TheArraysFormAValidCSRMatrix)
        {
            const IntegerCsr m = integer_csr(20, 15, 3, false);

            EXPECT_GT(m.nnz(), int64_t{0});
            expect_valid_csr(m);
        }

        TEST(Small_IntegerCsr, AStoredDiagonalHoldsOnes)
        {
            expect_stored_diagonal_of_ones(integer_csr(6, 6, 5, true));
        }

        TEST(Small_IntegerCsr, TheDenseTwinHoldsEveryStoredValue)
        {
            expect_dense_holds_each_stored_value(integer_csr(8, 9, 7, false));
        }

        TEST(Small_IntegerCsr, OneSeedGivesOneMatrix)
        {
            const IntegerCsr first = integer_csr(10, 10, 9, false);
            const IntegerCsr second = integer_csr(10, 10, 9, false);

            EXPECT_EQ(first.offsets, second.offsets);
            EXPECT_EQ(first.columns, second.columns);
            EXPECT_EQ(first.values, second.values);
        }
    } // namespace Test
} // namespace SushiBLAS
