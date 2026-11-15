/**************************************************************************/
/* test_along.cpp                                                         */
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
 * @file test_along.cpp
 * @brief Tests the reference take_along and put_along against hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/index_kernels/reference/along.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ReferenceAlong, TakeAlongPicksOneElementPerRow)
        {
            const std::vector<double> values{1, 2, 3, 4, 5, 6};
            const std::vector<int64_t> index{2, 0};
            std::vector<double> picked(2, -1.0);

            Kernels::Reference::take_along<double, int64_t>(
                Kernels::MatrixSpan{2, 3, 3, 1}, values.data(), index.data(), 1, picked.data(), 1);

            EXPECT_EQ(picked, (std::vector<double>{3, 4}));
        }

        TEST(Small_ReferenceAlong, TakeAlongWritesZeroForAnIndexOutsideTheRow)
        {
            const std::vector<double> values{1, 2, 3, 4, 5, 6};
            const std::vector<int32_t> index{-1, 3};
            std::vector<double> picked(2, -1.0);

            Kernels::Reference::take_along<double, int32_t>(
                Kernels::MatrixSpan{2, 3, 3, 1}, values.data(), index.data(), 1, picked.data(), 1);

            EXPECT_EQ(picked, (std::vector<double>{0, 0}));
        }

        TEST(Small_ReferenceAlong, PutAlongWritesThePickedValueAtItsIndexAndZerosElsewhere)
        {
            const std::vector<double> picked{7, 8};
            const std::vector<int64_t> index{1, 2};
            std::vector<double> values(6, -1.0);

            Kernels::Reference::put_along<double, int64_t>(
                picked.data(), 1, index.data(), 1, Kernels::MatrixSpan{2, 3, 3, 1}, values.data());

            EXPECT_EQ(values, (std::vector<double>{0, 7, 0, 0, 0, 8}));
        }

        TEST(Small_ReferenceAlong, PutAlongZeroesTheRowOfAnIndexOutsideIt)
        {
            const std::vector<double> picked{7, 8};
            const std::vector<int32_t> index{-1, 3};
            std::vector<double> values(6, -1.0);

            Kernels::Reference::put_along<double, int32_t>(
                picked.data(), 1, index.data(), 1, Kernels::MatrixSpan{2, 3, 3, 1}, values.data());

            EXPECT_EQ(values, (std::vector<double>(6, 0.0)));
        }
    } // namespace Test
} // namespace SushiBLAS
