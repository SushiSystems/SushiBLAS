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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
        TEST(Small_SmallIntegers, EveryValueIsAnIntegerFromMinusThreeToThree)
        {
            const std::vector<double> values = small_integers<double>(1000, 7);

            ASSERT_EQ(values.size(), std::size_t{1000});
            for (double value : values)
            {
                EXPECT_EQ(value, std::floor(value));
                EXPECT_GE(value, -3.0);
                EXPECT_LE(value, 3.0);
            }
        }

        TEST(Small_SmallIntegers, OneSeedGivesOneSequenceAndAnotherSeedAnother)
        {
            EXPECT_EQ(small_integers<float>(64, 1), small_integers<float>(64, 1));
            EXPECT_NE(small_integers<float>(64, 1), small_integers<float>(64, 2));
        }

        TEST(Small_SmallIntegers, TheUnitDiagonalMatrixHasOnesOnItsDiagonal)
        {
            const std::vector<int32_t> matrix = unit_diagonal_matrix<int32_t>(5, 3);

            ASSERT_EQ(matrix.size(), std::size_t{25});
            for (int64_t i = 0; i < 5; ++i)
                EXPECT_EQ(matrix[static_cast<std::size_t>(i * 5 + i)], 1);
        }
    } // namespace Test
} // namespace SushiBLAS
