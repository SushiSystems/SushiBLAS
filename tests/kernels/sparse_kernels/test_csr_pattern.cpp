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
 * @file test_csr_pattern.cpp
 * @brief Tests that a CsrPattern reads its rows and columns under either index base.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_CsrPattern, AZeroBasedPatternReadsItsArraysAsStored)
        {
            const std::vector<int32_t> offsets{0, 2, 2, 3};
            const std::vector<int32_t> columns{1, 3, 0};
            const Kernels::CsrPattern<int32_t> pattern{3, 4, 3, 0, offsets.data(), columns.data()};

            EXPECT_EQ(pattern.row_begin(0), int64_t{0});
            EXPECT_EQ(pattern.row_end(0), int64_t{2});
            EXPECT_EQ(pattern.row_begin(1), pattern.row_end(1));
            EXPECT_EQ(pattern.column(1), int64_t{3});
            EXPECT_EQ(pattern.row_begin(3), int64_t{3});
        }

        TEST(Unit_CsrPattern, AOneBasedPatternSubtractsItsBase)
        {
            const std::vector<int64_t> offsets{1, 3, 3, 4};
            const std::vector<int64_t> columns{2, 4, 1};
            const Kernels::CsrPattern<int64_t> pattern{3, 4, 3, 1, offsets.data(), columns.data()};

            EXPECT_EQ(pattern.row_begin(0), int64_t{0});
            EXPECT_EQ(pattern.row_end(2), int64_t{3});
            EXPECT_EQ(pattern.column(0), int64_t{1});
            EXPECT_EQ(pattern.column(2), int64_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
