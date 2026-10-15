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
 * @file test_integer_csr.cpp
 * @brief Tests that integer_csr builds a valid CSR matrix of small integers and its dense twin.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_IntegerCsr, TheArraysFormAValidCSRMatrix)
        {
            const IntegerCsr m = integer_csr(20, 15, 3, false);

            ASSERT_EQ(m.offsets.size(), std::size_t{21});
            EXPECT_EQ(m.offsets.front(), int64_t{0});
            EXPECT_EQ(m.offsets.back(), m.nnz());
            EXPECT_GT(m.nnz(), int64_t{0});
            for (int64_t i = 0; i < m.rows; ++i)
            {
                for (int64_t e = m.offsets[static_cast<std::size_t>(i)];
                     e < m.offsets[static_cast<std::size_t>(i + 1)]; ++e)
                {
                    EXPECT_LT(m.columns[static_cast<std::size_t>(e)], m.cols);
                    if (e > m.offsets[static_cast<std::size_t>(i)])
                    {
                        EXPECT_LT(m.columns[static_cast<std::size_t>(e - 1)],
                                  m.columns[static_cast<std::size_t>(e)]);
                    }
                }
            }
        }

        TEST(Unit_IntegerCsr, AStoredDiagonalHoldsOnes)
        {
            const IntegerCsr m = integer_csr(6, 6, 5, true);
            const std::vector<double> dense = m.dense();

            for (int64_t i = 0; i < 6; ++i)
                EXPECT_EQ(dense[static_cast<std::size_t>(i * 6 + i)], 1.0);
        }

        TEST(Unit_IntegerCsr, TheDenseTwinHoldsEveryStoredValue)
        {
            const IntegerCsr m = integer_csr(8, 9, 7, false);
            const std::vector<double> dense = m.dense();

            double stored = 0.0;
            for (const double value : m.values)
            {
                EXPECT_EQ(value, std::floor(value));
                stored += value;
            }
            double total = 0.0;
            for (const double value : dense)
                total += value;
            EXPECT_EQ(total, stored);
        }

        TEST(Unit_IntegerCsr, OneSeedGivesOneMatrix)
        {
            EXPECT_EQ(integer_csr(10, 10, 9, false).columns, integer_csr(10, 10, 9, false).columns);
        }
    } // namespace Test
} // namespace SushiBLAS
