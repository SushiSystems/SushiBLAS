/**************************************************************************/
/* test_im2col_index.cpp                                                  */
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
 * @file test_im2col_index.cpp
 * @brief Tests the index math of one im2col matrix element against the nested loops that fill the matrix.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/spatial_kernels/im2col_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds the extents of one im2col matrix. */
        struct MatrixRow
        {
            int64_t batch;
            int64_t out_h;
            int64_t out_w;
            int64_t kernel_h;
            int64_t kernel_w;
            int64_t channels;
        };

        /** @brief Returns the matrix elements in storage order, listed by the loops that fill the matrix. */
        static std::vector<Indexing::Im2colCoord> listed_elements(const MatrixRow& matrix)
        {
            std::vector<Indexing::Im2colCoord> elements;
            int64_t row = 0;
            for (int64_t n = 0; n < matrix.batch; ++n)
            {
                for (int64_t h_o = 0; h_o < matrix.out_h; ++h_o)
                {
                    for (int64_t w_o = 0; w_o < matrix.out_w; ++w_o, ++row)
                    {
                        int64_t col = 0;
                        for (int64_t r = 0; r < matrix.kernel_h; ++r)
                        {
                            for (int64_t s = 0; s < matrix.kernel_w; ++s)
                            {
                                for (int64_t c = 0; c < matrix.channels; ++c, ++col)
                                    elements.push_back({row, col, {n, h_o, w_o}, {r, s, c}});
                            }
                        }
                    }
                }
            }
            return elements;
        }

        /** @brief Checks im2col_coord against the listing for every element. */
        static void expect_elements_follow_the_listing(const MatrixRow& matrix)
        {
            const int64_t patch_width = matrix.kernel_h * matrix.kernel_w * matrix.channels;
            const std::vector<Indexing::Im2colCoord> listed = listed_elements(matrix);
            for (std::size_t i = 0; i < listed.size(); ++i)
            {
                const Indexing::Im2colCoord at = Indexing::im2col_coord(
                    static_cast<int64_t>(i), patch_width, matrix.channels, matrix.kernel_w,
                    matrix.out_h, matrix.out_w);
                EXPECT_EQ(at.row, listed[i].row) << "flat " << i;
                EXPECT_EQ(at.col, listed[i].col) << "flat " << i;
                EXPECT_EQ(at.window.n, listed[i].window.n) << "flat " << i;
                EXPECT_EQ(at.window.h_o, listed[i].window.h_o) << "flat " << i;
                EXPECT_EQ(at.window.w_o, listed[i].window.w_o) << "flat " << i;
                EXPECT_EQ(at.tap.r, listed[i].tap.r) << "flat " << i;
                EXPECT_EQ(at.tap.s, listed[i].tap.s) << "flat " << i;
                EXPECT_EQ(at.tap.c, listed[i].tap.c) << "flat " << i;
            }
        }

        class Small_Im2colIndexRows : public ::testing::TestWithParam<MatrixRow>
        {
        };

        TEST_P(Small_Im2colIndexRows, Im2colCoordFollowsTheFillOrderOfTheMatrix)
        {
            expect_elements_follow_the_listing(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_Im2colIndexRows,
                                 ::testing::Values(MatrixRow{0, 2, 2, 2, 2, 3},
                                                   MatrixRow{1, 0, 2, 2, 2, 3},
                                                   MatrixRow{1, 1, 1, 1, 1, 1},
                                                   MatrixRow{2, 2, 3, 2, 2, 3},
                                                   MatrixRow{1, 3, 5, 3, 2, 1},
                                                   MatrixRow{3, 1, 2, 1, 3, 5},
                                                   MatrixRow{1, 2, 2, 5, 1, 2}));

        TEST(Small_Im2colIndex, Im2colCoordSplitsFlatIndex37OfA2x2WindowGridWithThreeChannels)
        {
            // patch_width 12: row 37 / 12 = 3 is window (0, 1, 1), col 37 % 12 = 1 is tap (0, 0, 1).
            const Indexing::Im2colCoord at = Indexing::im2col_coord(37, 12, 3, 2, 2, 2);

            EXPECT_EQ(at.row, 3);
            EXPECT_EQ(at.col, 1);
            EXPECT_EQ(at.window.n, 0);
            EXPECT_EQ(at.window.h_o, 1);
            EXPECT_EQ(at.window.w_o, 1);
            EXPECT_EQ(at.tap.r, 0);
            EXPECT_EQ(at.tap.s, 0);
            EXPECT_EQ(at.tap.c, 1);
        }
    }
}
