/**************************************************************************/
/* test_spatial_geometry.cpp                                              */
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
 * @file test_spatial_geometry.cpp
 * @brief Tests output_extent, the window span and the default ConvGeometry without a device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <SushiBLAS/geometry/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Spatial::output_extent;

        namespace
        {
            /** @brief Holds the arguments of one output_extent call and the extent it gives. */
            struct ExtentRow
            {
                std::int64_t input;     /**< The input extent. */
                std::int64_t pad_low;   /**< Padding before the input. */
                std::int64_t pad_high;  /**< Padding after the input. */
                std::int64_t dilation;  /**< The kernel dilation. */
                std::int64_t kernel;    /**< The kernel extent. */
                std::int64_t stride;    /**< The stride. */
                std::int64_t expected;  /**< The hand extent. */
            };

            /**
             * @brief Returns the ordinary cases of the formula.
             * @details ResNet-18's stem 224 to 112 with a 7x7 kernel, stride 2 and padding 3; a 3x3 stride-1
             * "same" convolution keeps its extent; stride 2 halves it rounding down; no padding shrinks it by
             * the kernel; a default ConvGeometry maps every extent to itself.
             */
            [[nodiscard]] std::vector<ExtentRow> ordinary_rows()
            {
                const Spatial::ConvGeometry defaults;
                return {{224, 3, 3, 1, 7, 2, 112},
                        {32, 1, 1, 1, 3, 1, 32},
                        {32, 1, 1, 1, 3, 2, 16},
                        {32, 0, 0, 1, 3, 1, 30},
                        {17, 0, 0, defaults.dilation_h, defaults.kernel_h, defaults.stride_h, 17}};
            }

            /** @brief Returns the dilated cases: a dilated 3x3 covers 5 rows, so "same" needs 2 each side. */
            [[nodiscard]] std::vector<ExtentRow> dilation_rows()
            {
                return {{32, 2, 2, 2, 3, 1, 32}, {32, 0, 0, 2, 3, 1, 28}, {32, 0, 0, 4, 3, 1, 24}};
            }

            /** @brief Returns padding on the bottom only, on the top only and none, for a 2x2 stride-1 window. */
            [[nodiscard]] std::vector<ExtentRow> padding_rows()
            {
                return {{13, 0, 1, 1, 2, 1, 13}, {13, 1, 0, 1, 2, 1, 13}, {13, 0, 0, 1, 2, 1, 12}};
            }

            /** @brief Returns a window wider than the padded input, by size and by dilation. */
            [[nodiscard]] std::vector<ExtentRow> empty_rows()
            {
                return {{3, 0, 0, 1, 5, 1, 0}, {3, 0, 0, 4, 3, 1, 0}};
            }

            /** @brief Returns the window span the count tests read. */
            [[nodiscard]] Spatial::WindowSpan counted_span()
            {
                Spatial::WindowSpan span;
                span.batch = 2;
                span.channels = 3;
                span.out_h = 4;
                span.out_w = 5;
                span.geometry.kernel_h = 3;
                span.geometry.kernel_w = 3;
                return span;
            }
        } // namespace

        /** @brief Names the fixture of the rows an ordinary window gives. */
        class Small_SpatialGeometryOrdinary : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_SpatialGeometryOrdinary, TheOrdinaryCasesMatchTheFormula)
        {
            const ExtentRow& row = GetParam();
            EXPECT_EQ(output_extent(row.input, row.pad_low, row.pad_high, row.dilation, row.kernel, row.stride),
                      row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SpatialGeometryOrdinary, ::testing::ValuesIn(ordinary_rows()));

        /** @brief Names the fixture of the rows a dilated window gives. */
        class Small_SpatialGeometryDilation : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_SpatialGeometryDilation, DilationWidensTheKernelWithoutWideningTheOutput)
        {
            const ExtentRow& row = GetParam();
            EXPECT_EQ(output_extent(row.input, row.pad_low, row.pad_high, row.dilation, row.kernel, row.stride),
                      row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SpatialGeometryDilation, ::testing::ValuesIn(dilation_rows()));

        /** @brief Names the fixture of the rows a padded window gives. */
        class Small_SpatialGeometryPadding : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_SpatialGeometryPadding, PaddingOnEitherSideWidensThePaddedInputByItsAmount)
        {
            const ExtentRow& row = GetParam();
            EXPECT_EQ(output_extent(row.input, row.pad_low, row.pad_high, row.dilation, row.kernel, row.stride),
                      row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SpatialGeometryPadding, ::testing::ValuesIn(padding_rows()));

        /** @brief Names the fixture of the rows a window wider than its input gives. */
        class Small_SpatialGeometryEmpty : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_SpatialGeometryEmpty, AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne)
        {
            const ExtentRow& row = GetParam();
            EXPECT_EQ(output_extent(row.input, row.pad_low, row.pad_high, row.dilation, row.kernel, row.stride),
                      row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_SpatialGeometryEmpty, ::testing::ValuesIn(empty_rows()));

        TEST(Small_SpatialGeometry, WindowSpanCountsOneWindowPerOutputPosition)
        {
            EXPECT_EQ(counted_span().windows(), 40);
        }

        TEST(Small_SpatialGeometry, WindowSpanPatchHoldsTheChannelsOfOneWindow)
        {
            EXPECT_EQ(counted_span().patch_elements(), 27);
        }

        TEST(Small_SpatialGeometry, WindowSpanColumnHoldsEveryPatchOfEveryWindow)
        {
            EXPECT_EQ(counted_span().column_elements(), 1080);
        }
    } // namespace Test
} // namespace SushiBLAS
