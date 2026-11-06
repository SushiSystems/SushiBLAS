/**************************************************************************/
/* test_chunked_dot.cpp                                                   */
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
 * @file test_chunked_dot.cpp
 * @brief Tests that chunked_dot sums each chunk in index order and then the chunk sums in order.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/chunked_dot.hpp>

#include <vector>
#include <cstddef>
#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Two integer vectors and the exact value of their dot product. */
            struct IntegerDot
            {
                    std::vector<double> a;   /**< The left vector. */
                    std::vector<double> b;   /**< The right vector. */
                    double expected = 0.0;   /**< The dot product, exact in double. */
            };

            /** @brief Builds a and b from the residues of the index modulo 7 and 5, and their exact dot. */
            IntegerDot small_integer_dot(std::size_t n)
            {
                IntegerDot out;
                out.a.resize(n);
                out.b.resize(n);
                for (std::size_t i = 0; i < n; ++i)
                {
                    out.a[i] = static_cast<double>(i % 7) - 3.0;
                    out.b[i] = static_cast<double>(i % 5) - 2.0;
                    out.expected += out.a[i] * out.b[i];
                }
                return out;
            }
        } // namespace

        TEST(Small_ChunkedDot, CountsTheChunksOfALength)
        {
            EXPECT_EQ(Kernels::dot_chunks(0), 0u);
            EXPECT_EQ(Kernels::dot_chunks(1), 1u);
            EXPECT_EQ(Kernels::dot_chunks(Kernels::K_DOT_CHUNK), 1u);
            EXPECT_EQ(Kernels::dot_chunks(Kernels::K_DOT_CHUNK + 1), 2u);
        }

        TEST(Small_ChunkedDot, AddsTheChunkSumsRatherThanOneRunningSum)
        {
            // One running sum loses the 1 under 2^24; chunked, 2^24 - 2^24 cancels first.
            const std::size_t n = Kernels::K_DOT_CHUNK + 2;
            std::vector<float> a(n, 0.0f);
            const std::vector<float> ones(n, 1.0f);
            a[0] = 1.0f;
            a[Kernels::K_DOT_CHUNK] = 16777216.0f;
            a[Kernels::K_DOT_CHUNK + 1] = -16777216.0f;

            EXPECT_EQ(Kernels::chunk_dot(a.data(), ones.data(), n, 1), 0.0f);
            EXPECT_EQ(Kernels::chunked_dot(a.data(), ones.data(), n), 1.0f);
        }

        TEST(Small_ChunkedDot, GivesTheExactDotOfSmallIntegers)
        {
            const std::size_t n = 3 * Kernels::K_DOT_CHUNK + 7;
            const IntegerDot dot = small_integer_dot(n);

            EXPECT_EQ(Kernels::chunked_dot(dot.a.data(), dot.b.data(), n), dot.expected);
        }
    } // namespace Test
} // namespace SushiBLAS
