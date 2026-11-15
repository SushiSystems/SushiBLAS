/**************************************************************************/
/* test_gemm_split_regions.cpp                                            */
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
 * @file test_gemm_split_regions.cpp
 * @brief Tests gemm_part_regions() against a brute-force enumeration of the elements a part touches.
 * @author Mustafa Garip
 */

#include <set>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/gemm_split_regions.hpp>
#include <SushiRuntime/handle/global_handle.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Core::Layout;
            using Core::Transpose;
            using Dispatch::GemmGeometry;
            using Dispatch::GemmSplitAxis;

            /// @brief Bytes of one float element.
            constexpr std::size_t K_ELEMENT = sizeof(float);

            /// @brief Byte offset of the first element of C in its allocation.
            constexpr std::size_t K_C_OFFSET = 8;

            /** @brief Returns the operand starts of A, B and C, each in an allocation of its own. */
            Dispatch::GemmOperandRegions operand_starts()
            {
                Dispatch::GemmOperandRegions operands;
                operands.a = SushiRuntime::BufferRegion(SushiRuntime::GlobalHandle::from_words(7, 1), 0, 0);
                operands.b = SushiRuntime::BufferRegion(SushiRuntime::GlobalHandle::from_words(7, 2), 0, 0);
                operands.c = SushiRuntime::BufferRegion(SushiRuntime::GlobalHandle::from_words(7, 3), K_C_OFFSET, 0);
                return operands;
            }

            /** @brief Returns a batched geometry whose matrices sit stride elements apart. */
            GemmGeometry batched_geometry(Layout layout, Transpose trans_b, int64_t m, int64_t n,
                                          int64_t k, int64_t batch_size, int64_t gap)
            {
                const bool row_major = layout == Layout::RowMajor;
                const bool b_swapped = Core::is_transposed(trans_b);
                GemmGeometry geometry;
                geometry.m = m;
                geometry.n = n;
                geometry.k = k;
                geometry.batch_size = batch_size;
                geometry.a.leading_dimension = row_major ? k : m;
                geometry.b.leading_dimension = row_major ? (b_swapped ? k : n) : (b_swapped ? n : k);
                geometry.c.leading_dimension = row_major ? n : m;
                geometry.stride_a = m * k + gap;
                geometry.stride_b = k * n + gap;
                geometry.stride_c = m * n + gap;
                return geometry;
            }

            /** @brief Returns every byte the ranges of regions cover. */
            std::set<std::size_t> covered_bytes(const std::vector<SushiRuntime::BufferRegion>& regions,
                                                const SushiRuntime::GlobalHandle& handle)
            {
                std::set<std::size_t> bytes;
                for (const SushiRuntime::BufferRegion& region : regions)
                    for (std::size_t byte = region.offset; region.base == handle && byte < region.offset + region.length; ++byte)
                        bytes.insert(byte);
                return bytes;
            }

            /** @brief Returns every byte of the elements of C in rows [row_begin, row_end) and columns [column_begin, column_end) of every matrix. */
            std::set<std::size_t> element_bytes_of_c(const GemmGeometry& geometry, Layout layout,
                                                     int64_t row_begin, int64_t row_end,
                                                     int64_t column_begin, int64_t column_end)
            {
                std::set<std::size_t> bytes;
                for (int64_t matrix = 0; matrix < geometry.batch_size; ++matrix)
                    for (int64_t row = row_begin; row < row_end; ++row)
                        for (int64_t column = column_begin; column < column_end; ++column)
                        {
                            const int64_t element = matrix * geometry.stride_c +
                                (layout == Layout::RowMajor ? row * geometry.c.leading_dimension + column
                                                            : column * geometry.c.leading_dimension + row);
                            for (std::size_t byte = 0; byte < K_ELEMENT; ++byte)
                                bytes.insert(K_C_OFFSET + static_cast<std::size_t>(element) * K_ELEMENT + byte);
                        }
                return bytes;
            }

            /** @brief Returns every byte of the elements of op(B) in columns [column_begin, column_end) of every matrix. */
            std::set<std::size_t> element_bytes_of_op_b(const GemmGeometry& geometry, Layout layout,
                                                        Transpose trans_b, int64_t column_begin,
                                                        int64_t column_end)
            {
                const bool row_major = layout == Layout::RowMajor;
                const bool swapped = Core::is_transposed(trans_b);
                std::set<std::size_t> bytes;
                for (int64_t matrix = 0; matrix < geometry.batch_size; ++matrix)
                    for (int64_t row = 0; row < geometry.k; ++row)
                        for (int64_t column = column_begin; column < column_end; ++column)
                        {
                            const int64_t stored_row = swapped ? column : row;
                            const int64_t stored_column = swapped ? row : column;
                            const int64_t element = matrix * geometry.stride_b +
                                (row_major ? stored_row * geometry.b.leading_dimension + stored_column
                                           : stored_column * geometry.b.leading_dimension + stored_row);
                            for (std::size_t byte = 0; byte < K_ELEMENT; ++byte)
                                bytes.insert(static_cast<std::size_t>(element) * K_ELEMENT + byte);
                        }
                return bytes;
            }

            /** @brief Returns whether no two ranges of one handle touch, so the list is coalesced. */
            bool is_coalesced(const std::vector<SushiRuntime::BufferRegion>& regions)
            {
                for (std::size_t index = 1; index < regions.size(); ++index)
                    if (regions[index].base == regions[index - 1].base &&
                        regions[index - 1].offset + regions[index - 1].length >= regions[index].offset)
                        return false;
                return true;
            }
        } // namespace

        /** @brief Tests that a column part of a row-major C writes exactly the elements of its columns, one range per row. */
        TEST(Small_GemmPartRegions, AColumnPartOfARowMajorCWritesExactlyItsColumns)
        {
            const GemmGeometry geometry = batched_geometry(Layout::RowMajor, Transpose::NoTrans, 3, 7, 2, 1, 0);
            const Dispatch::GemmOperandRegions operands = operand_starts();

            const SushiRuntime::SplitRegions regions = Dispatch::gemm_part_regions(
                geometry, Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, false,
                GemmSplitAxis::Columns, 2, 5, K_ELEMENT, operands);

            EXPECT_EQ(covered_bytes(regions.writes, operands.c.base),
                      element_bytes_of_c(geometry, Layout::RowMajor, 0, 3, 2, 5));
            EXPECT_EQ(regions.writes.size(), 3u);
        }

        /** @brief Holds one layout and transpose of B a column part is read under. */
        struct OpBRow
        {
            Core::Layout layout;      /**< The layout of A, B and C. */
            Core::Transpose trans_b;  /**< How B is read. */
        };

        /** @brief Fixture of the rows of AColumnPartReadsExactlyItsColumnsOfOpB. */
        class Small_GemmPartRegionsOfB : public ::testing::TestWithParam<OpBRow>
        {
        };

        /** @brief Tests that a column part of a batched GEMM reads exactly its columns of op(B). */
        TEST_P(Small_GemmPartRegionsOfB, AColumnPartReadsExactlyItsColumnsOfOpB)
        {
            const OpBRow row = GetParam();
            const GemmGeometry geometry = batched_geometry(row.layout, row.trans_b, 3, 7, 4, 2, 5);
            const Dispatch::GemmOperandRegions operands = operand_starts();

            const SushiRuntime::SplitRegions regions = Dispatch::gemm_part_regions(
                geometry, row.layout, Transpose::NoTrans, row.trans_b, false,
                GemmSplitAxis::Columns, 1, 6, K_ELEMENT, operands);

            EXPECT_EQ(covered_bytes(regions.reads, operands.b.base),
                      element_bytes_of_op_b(geometry, row.layout, row.trans_b, 1, 6));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_GemmPartRegionsOfB,
            ::testing::Values(OpBRow{Core::Layout::RowMajor, Core::Transpose::NoTrans},
                              OpBRow{Core::Layout::RowMajor, Core::Transpose::Trans},
                              OpBRow{Core::Layout::ColumnMajor, Core::Transpose::NoTrans},
                              OpBRow{Core::Layout::ColumnMajor, Core::Transpose::Trans}));

        /** @brief Tests that a row part of a batched GEMM writes exactly its rows of every matrix, coalesced per matrix. */
        TEST(Small_GemmPartRegions, ARowPartOfABatchedCWritesItsRowsOfEveryMatrix)
        {
            const GemmGeometry geometry = batched_geometry(Layout::RowMajor, Transpose::NoTrans, 6, 4, 3, 3, 2);
            const Dispatch::GemmOperandRegions operands = operand_starts();

            const SushiRuntime::SplitRegions regions = Dispatch::gemm_part_regions(
                geometry, Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, false,
                GemmSplitAxis::Rows, 1, 4, K_ELEMENT, operands);

            EXPECT_EQ(covered_bytes(regions.writes, operands.c.base),
                      element_bytes_of_c(geometry, Layout::RowMajor, 1, 4, 0, 4));
            EXPECT_EQ(regions.writes.size(), 3u);
            EXPECT_TRUE(is_coalesced(regions.writes));
        }

        /** @brief Tests that a column part of a batched column-major C writes its columns of every matrix. */
        TEST(Small_GemmPartRegions, AColumnPartOfABatchedColumnMajorCWritesItsColumnsOfEveryMatrix)
        {
            const GemmGeometry geometry = batched_geometry(Layout::ColumnMajor, Transpose::NoTrans, 3, 7, 2, 2, 0);
            const Dispatch::GemmOperandRegions operands = operand_starts();

            const SushiRuntime::SplitRegions regions = Dispatch::gemm_part_regions(
                geometry, Layout::ColumnMajor, Transpose::NoTrans, Transpose::NoTrans, false,
                GemmSplitAxis::Columns, 2, 5, K_ELEMENT, operands);

            EXPECT_EQ(covered_bytes(regions.writes, operands.c.base),
                      element_bytes_of_c(geometry, Layout::ColumnMajor, 0, 3, 2, 5));
            EXPECT_TRUE(is_coalesced(regions.writes));
        }

        /** @brief Tests that a part reads its block of C only when beta asks for the old C. */
        TEST(Small_GemmPartRegions, APartReadsCOnlyWhenBetaReadsIt)
        {
            const GemmGeometry geometry = batched_geometry(Layout::RowMajor, Transpose::NoTrans, 3, 7, 2, 1, 0);
            const Dispatch::GemmOperandRegions operands = operand_starts();

            const SushiRuntime::SplitRegions reading = Dispatch::gemm_part_regions(
                geometry, Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, true,
                GemmSplitAxis::Columns, 2, 5, K_ELEMENT, operands);
            const SushiRuntime::SplitRegions writing = Dispatch::gemm_part_regions(
                geometry, Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, false,
                GemmSplitAxis::Columns, 2, 5, K_ELEMENT, operands);

            EXPECT_EQ(covered_bytes(reading.reads, operands.c.base), covered_bytes(reading.writes, operands.c.base));
            EXPECT_TRUE(covered_bytes(writing.reads, operands.c.base).empty());
        }
    } // namespace Test
} // namespace SushiBLAS
