/**************************************************************************/
/* test_packed_product_harness.cpp                                        */
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
 * @file test_packed_product_harness.cpp
 * @brief Tests the harness helpers: the patterned values, the blockings and the problem it builds.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_PackedProductHarness, PatternedElementsAreSmallIntegers)
        {
            for (int64_t row = 0; row < 10; ++row)
            {
                for (int64_t col = 0; col < 10; ++col)
                {
                    const double value = patterned_element<double>(row, col, 4);
                    EXPECT_EQ(value, std::floor(value));
                    EXPECT_GE(value, -3.0);
                    EXPECT_LE(value, 3.0);
                }
            }
            EXPECT_NE(patterned_element<float>(1, 2, 0), patterned_element<float>(1, 2, 1));
        }

        TEST(Medium_PackedProductHarness, PoisonIsNaNAndCountsAsAMismatch)
        {
            const std::vector<float> want{1.0f, 2.0f};
            const std::vector<float> got{1.0f, poison_element<float>()};

            EXPECT_TRUE(std::isnan(poison_element<double>()));
            EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), 2), std::size_t{1});
            EXPECT_EQ(count_bit_mismatches(want.data(), want.data(), 2), std::size_t{0});
        }

        TEST(Medium_PackedProductHarness, TheBlockingsSpanTheExtremes)
        {
            EXPECT_EQ(smallest_blocking().inner_bytes, std::size_t{1});
            EXPECT_EQ(widest_blocking().outer_bytes, std::numeric_limits<std::size_t>::max());
            EXPECT_GT(ragged_blocking<float>().inner_bytes, smallest_blocking().inner_bytes);
        }

        /** @brief Builds problems on the host CPU queue. */
        class Medium_PackedProductHarnessOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_PackedProductHarnessOnDevice, TheProblemsKernelsAgreeWithItsExpectation)
        {
            GemmProblem<float> problem(queue(), Core::Layout::ColumnMajor,
                                       Core::Transpose::Trans, Core::Transpose::NoTrans, 13, 11,
                                       9, 2);

            const std::vector<float> expected = problem.expected(1.0f, 1.0f);
            ASSERT_EQ(expected.size(), problem.output_size());
            EXPECT_EQ(problem.unpacked(1.0f, 1.0f), expected);
            EXPECT_EQ(problem.packed(ragged_blocking<float>(), 2, 1.0f, 1.0f), expected);
        }
    } // namespace Test
} // namespace SushiBLAS
