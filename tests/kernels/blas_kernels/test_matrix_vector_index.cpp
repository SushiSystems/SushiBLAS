/**************************************************************************/
/* test_matrix_vector_index.cpp                                           */
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

/**
 * @file test_matrix_vector_index.cpp
 * @brief Tests the extent, row-shape and row-window math of the matrix-vector kernels against hand-derived values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/matrix_vector_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds a matrix shape, whether op(A) transposes it, and the lengths of y and x. */
            struct ExtentRow
            {
                bool transposed;
                int64_t m;
                int64_t n;
                int64_t out_len;
                int64_t in_len;
            };

            /** @brief Checks the lengths vector_extents gives for the row. */
            void expect_extents(const ExtentRow& row)
            {
                const Indexing::VectorExtents got =
                    Indexing::vector_extents(row.transposed, row.m, row.n);
                EXPECT_EQ(got.out_len, row.out_len);
                EXPECT_EQ(got.in_len, row.in_len);
            }
        } // namespace

        class Small_MatrixVectorIndexExtents : public ::testing::TestWithParam<ExtentRow>
        {
        };

        TEST_P(Small_MatrixVectorIndexExtents, TransposingSwapsTheLengthsOfYAndX)
        {
            expect_extents(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Small_MatrixVectorIndexExtents,
                                 ::testing::Values(ExtentRow{false, 3, 5, 3, 5},
                                                   ExtentRow{true, 3, 5, 5, 3},
                                                   ExtentRow{false, 0, 9, 0, 9},
                                                   ExtentRow{true, 9, 0, 0, 9},
                                                   ExtentRow{false, 1, 1, 1, 1},
                                                   ExtentRow{true, 2, 2, 2, 2}));

        /** @brief Holds a layout, a transpose and whether a row of op(A) is then contiguous. */
        struct ContiguousRow
        {
            bool row_major;
            bool transposed;
            bool contiguous;
        };

        class Small_MatrixVectorIndexContiguous : public ::testing::TestWithParam<ContiguousRow>
        {
        };

        TEST_P(Small_MatrixVectorIndexContiguous, ARowIsContiguousWhenTheTransposeUndoesTheLayout)
        {
            EXPECT_EQ(Indexing::rows_are_contiguous(GetParam().row_major, GetParam().transposed),
                      GetParam().contiguous);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_MatrixVectorIndexContiguous,
                                 ::testing::Values(ContiguousRow{true, false, true},
                                                   ContiguousRow{true, true, false},
                                                   ContiguousRow{false, false, false},
                                                   ContiguousRow{false, true, true}));

        /** @brief Holds a row count, the units and group size of a device, and the verdict. */
        struct UnderfillRow
        {
            int64_t rows;
            std::size_t units;
            std::size_t group_size;
            bool underfills;
        };

        class Small_MatrixVectorIndexUnderfill : public ::testing::TestWithParam<UnderfillRow>
        {
        };

        TEST_P(Small_MatrixVectorIndexUnderfill, RowsUnderfillBelowOneGroupPerUnit)
        {
            const UnderfillRow row = GetParam();
            EXPECT_EQ(Indexing::rows_underfill_device(row.rows, row.units, row.group_size),
                      row.underfills);
        }

        // Four units of eight items need 32 rows; 31 underfill, 32 and 33 do not.
        INSTANTIATE_TEST_SUITE_P(, Small_MatrixVectorIndexUnderfill,
                                 ::testing::Values(UnderfillRow{0, 4, 8, true},
                                                   UnderfillRow{31, 4, 8, true},
                                                   UnderfillRow{32, 4, 8, false},
                                                   UnderfillRow{33, 4, 8, false},
                                                   UnderfillRow{0, 1, 1, true},
                                                   UnderfillRow{1, 1, 1, false}));

        /** @brief Holds an extent, a band and the longest row it gives. */
        struct BandRow
        {
            int64_t extent;
            int64_t lower;
            int64_t upper;
            int64_t expected;
        };

        class Small_MatrixVectorIndexBand : public ::testing::TestWithParam<BandRow>
        {
        };

        TEST_P(Small_MatrixVectorIndexBand, ABandRowHoldsTheBandWidthAtMostTheExtent)
        {
            const BandRow row = GetParam();
            EXPECT_EQ(Indexing::band_row_length(row.extent, row.lower, row.upper), row.expected);
        }

        // The band width is lower + upper + 1: 3 for (1, 1), 9 for (4, 4), 1 for (0, 0).
        INSTANTIATE_TEST_SUITE_P(, Small_MatrixVectorIndexBand,
                                 ::testing::Values(BandRow{9, 1, 1, 3}, BandRow{2, 1, 1, 2},
                                                   BandRow{5, 4, 4, 5}, BandRow{0, 2, 2, 0},
                                                   BandRow{9, 0, 0, 1}, BandRow{3, 2, 0, 3},
                                                   BandRow{1, 0, 5, 1}));

        /** @brief Holds a row length, a row and the whole-row window it gets. */
        struct WholeRowRow
        {
            int64_t length;
            int64_t row;
            int64_t first;
            int64_t last;
        };

        class Small_MatrixVectorIndexWholeRow : public ::testing::TestWithParam<WholeRowRow>
        {
        };

        TEST_P(Small_MatrixVectorIndexWholeRow, EveryRowGetsEveryColumn)
        {
            const WholeRowRow row = GetParam();
            const Indexing::RowWindow window = Indexing::WholeRow{row.length}(row.row);
            EXPECT_EQ(window.first, row.first);
            EXPECT_EQ(window.last, row.last);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_MatrixVectorIndexWholeRow,
                                 ::testing::Values(WholeRowRow{0, 0, 0, 0}, WholeRowRow{1, 0, 0, 1},
                                                   WholeRowRow{9, 5, 0, 9}, WholeRowRow{100, 3, 0, 100}));

        /** @brief Holds a triangle, a row and the columns it stores. */
        struct TriangleWindowRow
        {
            int64_t order;
            bool upper;
            int64_t row;
            int64_t first;
            int64_t last;
        };

        class Small_MatrixVectorIndexTriangleRow : public ::testing::TestWithParam<TriangleWindowRow>
        {
        };

        TEST_P(Small_MatrixVectorIndexTriangleRow, AnUpperRowStartsAtTheDiagonalALowerRowEndsAtIt)
        {
            const TriangleWindowRow row = GetParam();
            const Indexing::RowWindow window = Indexing::TriangleRow{row.order, row.upper}(row.row);
            EXPECT_EQ(window.first, row.first);
            EXPECT_EQ(window.last, row.last);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_MatrixVectorIndexTriangleRow,
                                 ::testing::Values(TriangleWindowRow{1, true, 0, 0, 1},
                                                   TriangleWindowRow{1, false, 0, 0, 1},
                                                   TriangleWindowRow{9, true, 0, 0, 9},
                                                   TriangleWindowRow{9, true, 3, 3, 9},
                                                   TriangleWindowRow{9, true, 8, 8, 9},
                                                   TriangleWindowRow{9, false, 0, 0, 1},
                                                   TriangleWindowRow{9, false, 3, 0, 4},
                                                   TriangleWindowRow{9, false, 8, 0, 9},
                                                   TriangleWindowRow{10, true, 3, 3, 10},
                                                   TriangleWindowRow{10, false, 3, 0, 4}));
    } // namespace Test
} // namespace SushiBLAS
