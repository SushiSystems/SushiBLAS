/**************************************************************************/
/* test_small_integers.cpp                                                */
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
 * @file test_small_integers.cpp
 * @brief Tests that the operands are small integers, repeat for a seed and keep a unit diagonal.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Checks that every value is an integer from -3 to 3. */
            void expect_small_integers(const std::vector<double>& values)
            {
                for (double value : values)
                {
                    EXPECT_EQ(value, std::floor(value));
                    EXPECT_GE(value, -3.0);
                    EXPECT_LE(value, 3.0);
                }
            }

            /** @brief Checks that every diagonal element of an n by n row-major matrix is 1. */
            void expect_unit_diagonal(const std::vector<int32_t>& matrix, int64_t n)
            {
                for (int64_t i = 0; i < n; ++i)
                    EXPECT_EQ(matrix[static_cast<std::size_t>(i * n + i)], 1);
            }
        } // namespace

        TEST(Small_SmallIntegers, EveryValueIsAnIntegerFromMinusThreeToThree)
        {
            const std::vector<double> values = small_integers<double>(1000, 7);

            ASSERT_EQ(values.size(), std::size_t{1000});
            expect_small_integers(values);
        }

        TEST(Small_SmallIntegers, OneSeedRepeatsItsSequence)
        {
            EXPECT_EQ(small_integers<float>(64, 1), small_integers<float>(64, 1));
        }

        TEST(Small_SmallIntegers, AnotherSeedGivesAnotherSequence)
        {
            EXPECT_NE(small_integers<float>(64, 1), small_integers<float>(64, 2));
        }

        TEST(Small_SmallIntegers, TheUnitDiagonalMatrixHasOnesOnItsDiagonal)
        {
            const std::vector<int32_t> matrix = unit_diagonal_matrix<int32_t>(5, 3);

            ASSERT_EQ(matrix.size(), std::size_t{25});
            expect_unit_diagonal(matrix, 5);
        }
    } // namespace Test
} // namespace SushiBLAS
