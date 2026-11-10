/**************************************************************************/
/* test_blend_index.cpp                                                   */
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
 * @file test_blend_index.cpp
 * @brief Tests the index math and the homogeneous-point step of the transform blend against nested loops and hand values.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blend_kernels/blend_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        /** @brief Holds the extents of a transform buffer `[T][rows][4]`. */
        struct TransformRow
        {
            int64_t count;
            int64_t rows;
        };

        /** @brief Checks that matrix_offset plus matrix_element visits the buffer in storage order. */
        static void expect_matrix_offsets_are_sequential(const TransformRow& shape)
        {
            int64_t expected = 0;
            for (int64_t slot = 0; slot < shape.count; ++slot)
            {
                for (int64_t row = 0; row < shape.rows; ++row)
                {
                    for (int64_t col = 0; col < 4; ++col, ++expected)
                    {
                        EXPECT_EQ(Indexing::matrix_offset(slot, shape.rows) +
                                      Indexing::matrix_element(row, col),
                                  expected)
                            << "slot " << slot << " row " << row << " col " << col;
                    }
                }
            }
        }

        class Small_BlendIndexTransforms : public ::testing::TestWithParam<TransformRow>
        {
        };

        TEST_P(Small_BlendIndexTransforms, MatrixOffsetsWalkThePackedTransformBuffer)
        {
            expect_matrix_offsets_are_sequential(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BlendIndexTransforms,
                                 ::testing::Values(TransformRow{0, 3}, TransformRow{1, 3},
                                                   TransformRow{2, 4}, TransformRow{3, 3},
                                                   TransformRow{5, 4}, TransformRow{9, 3}));

        /** @brief Holds the extents of the point and influence buffers. */
        struct PointRow
        {
            int64_t points;
            int64_t components;
            int64_t influences;
        };

        /** @brief Checks that point_offset and influence_offset visit their buffers in storage order. */
        static void expect_point_offsets_are_sequential(const PointRow& shape)
        {
            int64_t component_cursor = 0;
            int64_t influence_cursor = 0;
            for (int64_t p = 0; p < shape.points; ++p)
            {
                EXPECT_EQ(Indexing::point_offset(p, shape.components), component_cursor);
                component_cursor += shape.components;
                for (int64_t j = 0; j < shape.influences; ++j, ++influence_cursor)
                {
                    EXPECT_EQ(Indexing::influence_offset(p, shape.influences, j),
                              influence_cursor);
                }
            }
        }

        class Small_BlendIndexPoints : public ::testing::TestWithParam<PointRow>
        {
        };

        TEST_P(Small_BlendIndexPoints, PointAndInfluenceOffsetsWalkTheirBuffers)
        {
            expect_point_offsets_are_sequential(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BlendIndexPoints,
                                 ::testing::Values(PointRow{0, 3, 4}, PointRow{1, 3, 1},
                                                   PointRow{2, 4, 2}, PointRow{3, 3, 4},
                                                   PointRow{5, 4, 1}, PointRow{9, 3, 3}));

        namespace
        {
            /** @brief Holds one slot_live call. */
            struct SlotRow
            {
                int64_t slot;
                int64_t transform_count;
                bool expected;
            };
        } // namespace

        class Small_BlendIndexSlots : public ::testing::TestWithParam<SlotRow>
        {
        };

        TEST_P(Small_BlendIndexSlots, SlotLiveIsTrueOnlyInsideTheTransformCount)
        {
            const SlotRow row = GetParam();

            EXPECT_EQ(Indexing::slot_live(row.slot, row.transform_count), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_BlendIndexSlots,
                                 ::testing::Values(SlotRow{-1, 3, false}, SlotRow{0, 3, true},
                                                   SlotRow{2, 3, true}, SlotRow{3, 3, false},
                                                   SlotRow{0, 0, false}, SlotRow{0, 1, true},
                                                   SlotRow{1, 1, false}));

        /** @brief Holds one has_matrix_row call. */
        struct RowRow
        {
            int64_t component;
            int64_t transform_rows;
            bool expected;
        };

        class Small_BlendIndexRows : public ::testing::TestWithParam<RowRow>
        {
        };

        TEST_P(Small_BlendIndexRows, HasMatrixRowIsFalseOnlyForTheImpliedLastRow)
        {
            const RowRow row = GetParam();

            EXPECT_EQ(Indexing::has_matrix_row(row.component, row.transform_rows), row.expected);
        }

        // With 3 stored rows the fourth component is the implied (0, 0, 0, 1); with 4 it is stored.
        INSTANTIATE_TEST_SUITE_P(, Small_BlendIndexRows,
                                 ::testing::Values(RowRow{0, 3, true}, RowRow{2, 3, true},
                                                   RowRow{3, 3, false}, RowRow{3, 4, true},
                                                   RowRow{0, 0, false}));

        TEST(Small_BlendIndex, HomogeneousWIsTheFourthValueOfAHomogeneousPoint)
        {
            const float point[4] = {1.0f, 2.0f, 3.0f, 0.5f};

            EXPECT_EQ(Indexing::homogeneous_w<double>(point, 4), 0.5);
        }

        TEST(Small_BlendIndex, HomogeneousWIsOneForAPosition)
        {
            const float point[3] = {1.0f, 2.0f, 3.0f};

            EXPECT_EQ(Indexing::homogeneous_w<double>(point, 3), 1.0);
        }
    }
}
