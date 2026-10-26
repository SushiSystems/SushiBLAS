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
            std::vector<double> a(n);
            std::vector<double> b(n);
            double expected = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                a[i] = static_cast<double>(i % 7) - 3.0;
                b[i] = static_cast<double>(i % 5) - 2.0;
                expected += a[i] * b[i];
            }

            EXPECT_EQ(Kernels::chunked_dot(a.data(), b.data(), n), expected);
        }
    } // namespace Test
} // namespace SushiBLAS
