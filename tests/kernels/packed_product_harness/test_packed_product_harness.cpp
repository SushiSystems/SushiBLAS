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
 * @file test_packed_product_harness.cpp
 * @brief Tests the harness helpers: the patterned values, the blockings and the problem it builds.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
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
        /// @brief One position, one salt and the element the pattern puts there.
        struct PatternRow
        {
                const char* name;
                int64_t row;
                int64_t col;
                int64_t salt;
                double expected; ///< Worked by hand from (3 * row + 5 * col + salt) % 7 - 3.
        };

        /// @brief Returns positions that reach both ends of the range [-3, 3].
        static std::vector<PatternRow> pattern_rows()
        {
            return {{"Origin_LowestValue", 0, 0, 0, -3.0},
                    {"SecondColumn", 0, 1, 0, 2.0},
                    {"SecondRow", 1, 0, 0, 0.0},
                    {"Row1Column2_HighestValue", 1, 2, 0, 3.0},
                    {"Row2Column3Salt4", 2, 3, 4, 1.0},
                    {"Row1Column2Salt1", 1, 2, 1, -3.0}};
        }

        /// @brief Names a row, which gtest takes as the test's suffix.
        static std::string pattern_row_name(const ::testing::TestParamInfo<PatternRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs the pattern at one position.
        class Small_PackedProductHarnessPattern : public ::testing::TestWithParam<PatternRow>
        {
        };

        TEST_P(Small_PackedProductHarnessPattern, PatternedElementsAreSmallIntegers)
        {
            const PatternRow& row = GetParam();

            EXPECT_EQ(patterned_element<double>(row.row, row.col, row.salt), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_PackedProductHarnessPattern,
                                 ::testing::ValuesIn(pattern_rows()), pattern_row_name);

        TEST(Small_PackedProductHarness, TheSaltMovesThePatternAtOnePosition)
        {
            EXPECT_NE(patterned_element<float>(1, 2, 0), patterned_element<float>(1, 2, 1));
        }

        TEST(Small_PackedProductHarness, PoisonIsNaN)
        {
            EXPECT_TRUE(std::isnan(poison_element<float>()));
            EXPECT_TRUE(std::isnan(poison_element<double>()));
        }

        TEST(Small_PackedProductHarness, AMismatchIsCountedWhereTheBitsDiffer)
        {
            const std::vector<float> want{1.0f, 2.0f, 0.0f};
            const std::vector<float> got{1.0f, poison_element<float>(), -0.0f};

            EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), 3), std::size_t{2});
            EXPECT_EQ(count_bit_mismatches(want.data(), want.data(), 3), std::size_t{0});
        }

        TEST(Small_PackedProductHarness, TheBlockingsSpanTheExtremes)
        {
            EXPECT_EQ(smallest_blocking().inner_bytes, std::size_t{1});
            EXPECT_EQ(widest_blocking().outer_bytes, std::numeric_limits<std::size_t>::max());
            EXPECT_GT(ragged_blocking<float>().inner_bytes, smallest_blocking().inner_bytes);
        }

        TEST(Small_PackedProductHarness, ALeadingDimensionExceedsTheContiguousExtentByThePadding)
        {
            // Column-major keeps columns contiguous, so the rows are the extent; row-major, the columns.
            EXPECT_EQ(padded_leading_dimension(Core::Layout::ColumnMajor, 13, 11),
                      13 + K_PACKED_TEST_LD_PADDING);
            EXPECT_EQ(padded_leading_dimension(Core::Layout::RowMajor, 13, 11),
                      11 + K_PACKED_TEST_LD_PADDING);
        }

        TEST(Small_PackedProductHarness, ABatchStrideHoldsOneMatrixAndTheGap)
        {
            // Column-major: ld 16 for each of 11 columns; row-major: ld 14 for each of 13 rows.
            EXPECT_EQ(padded_batch_stride(Core::Layout::ColumnMajor, 13, 11, 16),
                      16 * 11 + K_PACKED_TEST_BATCH_GAP);
            EXPECT_EQ(padded_batch_stride(Core::Layout::RowMajor, 13, 11, 14),
                      14 * 13 + K_PACKED_TEST_BATCH_GAP);
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
