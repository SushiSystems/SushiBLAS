/**************************************************************************/
/* test_row_block_index.cpp                                               */
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
 * @file test_row_block_index.cpp
 * @brief Tests the row-block counts, targets and boundaries of the scattering products against a scan and hand values.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/row_block_index.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Indexing = Kernels::Indexing;

        namespace
        {
            /** @brief Holds the arrays of a CSR matrix built from its row lengths. */
            struct OwnedCsr
            {
                    std::vector<int64_t> offsets;   /**< rows + 1 offsets. */
                    std::vector<int64_t> columns;   /**< One column per stored entry. */

                    /** @brief Returns the pattern over the arrays, base zero. */
                    Kernels::CsrPattern<int64_t> pattern() const
                    {
                        const int64_t rows = static_cast<int64_t>(offsets.size()) - 1;
                        const int64_t nnz = offsets.back();
                        return Kernels::CsrPattern<int64_t>{rows, 1, nnz, 0, offsets.data(),
                                                            columns.data()};
                    }
            };

            /** @brief Returns the arrays of a matrix whose row i stores lengths[i] entries. */
            OwnedCsr csr_of_row_lengths(const std::vector<int64_t>& lengths)
            {
                OwnedCsr csr;
                csr.offsets.push_back(0);
                for (const int64_t length : lengths)
                    csr.offsets.push_back(csr.offsets.back() + length);
                csr.columns.assign(static_cast<std::size_t>(csr.offsets.back()), 0);
                return csr;
            }

            /** @brief Returns the first row in [0, rows] whose entries start at or after target, by a scan. */
            int64_t first_row_by_scan(const Kernels::CsrPattern<int64_t>& pattern, int64_t target)
            {
                for (int64_t row = 0; row < pattern.rows; ++row)
                {
                    if (pattern.row_begin(row) >= target)
                        return row;
                }
                return pattern.rows;
            }

            /** @brief Holds a matrix as its row lengths and the number of blocks to cut it into. */
            struct BlockedRows
            {
                    std::vector<int64_t> lengths;   /**< Stored entries of each row. */
                    int64_t blocks;                 /**< Blocks to cut the entries into. */
            };

            /** @brief Checks block_first_row against the scan for every block boundary of a matrix. */
            void expect_first_rows_follow_the_scan(const BlockedRows& row)
            {
                const OwnedCsr csr = csr_of_row_lengths(row.lengths);
                const Kernels::CsrPattern<int64_t> pattern = csr.pattern();
                for (int64_t block = 0; block <= row.blocks; ++block)
                {
                    const int64_t target = Indexing::block_target(pattern.nnz, block, row.blocks);
                    EXPECT_EQ(Indexing::block_first_row(pattern, block, row.blocks),
                              first_row_by_scan(pattern, target))
                        << "block " << block;
                }
            }

            /** @brief Checks that the blocks start at row 0, end at the last row, and touch end to start. */
            void expect_blocks_cover_the_rows(const BlockedRows& row)
            {
                const OwnedCsr csr = csr_of_row_lengths(row.lengths);
                const Kernels::CsrPattern<int64_t> pattern = csr.pattern();
                EXPECT_EQ(Indexing::block_first_row(pattern, 0, row.blocks), int64_t{0});
                EXPECT_EQ(Indexing::block_end_row(pattern, row.blocks - 1, row.blocks), pattern.rows);
                for (int64_t block = 0; block + 1 < row.blocks; ++block)
                    EXPECT_EQ(Indexing::block_end_row(pattern, block, row.blocks),
                              Indexing::block_first_row(pattern, block + 1, row.blocks));
            }
        } // namespace

        class Small_RowBlockIndexRows : public ::testing::TestWithParam<BlockedRows>
        {
        };

        TEST_P(Small_RowBlockIndexRows, BlockFirstRowIsTheFirstRowAtOrAfterTheBlockTarget)
        {
            expect_first_rows_follow_the_scan(GetParam());
        }

        TEST_P(Small_RowBlockIndexRows, TheBlocksCoverTheRowsInOrder)
        {
            expect_blocks_cover_the_rows(GetParam());
        }

        // No rows, empty rows, one row, uneven rows, and 1, 2, 3, 5 and 9 blocks.
        INSTANTIATE_TEST_SUITE_P(, Small_RowBlockIndexRows,
                                 ::testing::Values(BlockedRows{{}, 1},
                                                   BlockedRows{{0}, 2},
                                                   BlockedRows{{1}, 1},
                                                   BlockedRows{{1}, 3},
                                                   BlockedRows{{2, 0, 3}, 2},
                                                   BlockedRows{{2, 0, 3}, 5},
                                                   BlockedRows{{1, 1, 1, 1, 1}, 3},
                                                   BlockedRows{{0, 0, 4, 0, 0, 5, 0, 0, 0}, 9},
                                                   BlockedRows{{3, 3, 3, 3, 3, 3, 3, 3, 3}, 2}));

        /** @brief Holds one block_target call and the value its definition gives. */
        struct BlockTargetRow
        {
            int64_t nnz;
            int64_t block;
            int64_t blocks;
            int64_t expected;
        };

        class Small_RowBlockIndexTarget : public ::testing::TestWithParam<BlockTargetRow>
        {
        };

        TEST_P(Small_RowBlockIndexTarget, BlockTargetIsTheEntryCountTimesBlockOverBlocksRoundedDown)
        {
            const BlockTargetRow row = GetParam();

            EXPECT_EQ(Indexing::block_target(row.nnz, row.block, row.blocks), row.expected);
        }

        // floor(nnz * block / blocks): 10/3 = 3, 20/3 = 6, 10, 0, 0, 1 / 2 = 0, 1, 5 / 2 = 2, 1, 9, 6 / 5 = 1.
        INSTANTIATE_TEST_SUITE_P(, Small_RowBlockIndexTarget,
                                 ::testing::Values(BlockTargetRow{10, 0, 3, 0}, BlockTargetRow{10, 1, 3, 3},
                                                   BlockTargetRow{10, 2, 3, 6}, BlockTargetRow{10, 3, 3, 10},
                                                   BlockTargetRow{0, 2, 4, 0}, BlockTargetRow{1, 1, 2, 0},
                                                   BlockTargetRow{1, 2, 2, 1}, BlockTargetRow{5, 1, 2, 2},
                                                   BlockTargetRow{9, 1, 9, 1}, BlockTargetRow{9, 9, 9, 9},
                                                   BlockTargetRow{3, 2, 5, 1}));

        /** @brief Holds one scatter_block_count call and the count it must give. */
        struct ScatterCountRow
        {
            std::size_t resident;
            int64_t outputs;
            int64_t nnz;
            std::size_t expected;
        };

        class Small_RowBlockIndexCount : public ::testing::TestWithParam<ScatterCountRow>
        {
        };

        TEST_P(Small_RowBlockIndexCount, TheCountIsTheSmallestOfResidentAffordableAndTheMaximum)
        {
            const ScatterCountRow row = GetParam();

            EXPECT_EQ(Indexing::scatter_block_count(row.resident, row.outputs, row.nnz), row.expected);
        }

        // Affordable is 2 * nnz / outputs: 0, 2, 3, 3, 3, 2, 66 (resident 2 binds), 2000 (the maximum binds).
        INSTANTIATE_TEST_SUITE_P(, Small_RowBlockIndexCount,
                                 ::testing::Values(ScatterCountRow{8, 0, 100, 0},
                                                   ScatterCountRow{8, 1, 0, 1},
                                                   ScatterCountRow{8, 1, 1, 2},
                                                   ScatterCountRow{8, 2, 3, 3},
                                                   ScatterCountRow{8, 3, 5, 3},
                                                   ScatterCountRow{8, 5, 9, 3},
                                                   ScatterCountRow{8, 9, 9, 2},
                                                   ScatterCountRow{2, 3, 100, 2},
                                                   ScatterCountRow{1000, 1, 1000,
                                                                 Indexing::K_MAX_SCATTER_BLOCKS},
                                                   ScatterCountRow{0, 5, 100, 1}));

        /** @brief Holds one resident_blocks call. */
        struct ResidentBlocksRow
        {
            bool is_cpu;
            std::size_t compute_units;
            std::size_t waves_per_unit;
            std::size_t expected;
        };

        class Small_RowBlockIndexResident : public ::testing::TestWithParam<ResidentBlocksRow>
        {
        };

        TEST_P(Small_RowBlockIndexResident, AGpuKeepsUnitsTimesWavesAndACpuKeepsUnits)
        {
            const ResidentBlocksRow row = GetParam();

            EXPECT_EQ(Indexing::resident_blocks(row.is_cpu, row.compute_units, row.waves_per_unit),
                      row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RowBlockIndexResident,
                                 ::testing::Values(ResidentBlocksRow{true, 12, 8, 12},
                                                   ResidentBlocksRow{true, 0, 8, 0},
                                                   ResidentBlocksRow{false, 4, 8, 32},
                                                   ResidentBlocksRow{false, 0, 8, 0},
                                                   ResidentBlocksRow{false, 3, 1, 3}));

        /** @brief Holds one partial_index call: output j of slice `block`, slices `outputs` wide. */
        struct PartialIndexRow
        {
            int64_t block;
            int64_t outputs;
            int64_t j;
            int64_t expected;
        };

        class Small_RowBlockIndexPartial : public ::testing::TestWithParam<PartialIndexRow>
        {
        };

        TEST_P(Small_RowBlockIndexPartial, PartialIndexIsTheSliceStartPlusTheOutput)
        {
            const PartialIndexRow row = GetParam();

            EXPECT_EQ(Indexing::partial_index(row.block, row.outputs, row.j), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RowBlockIndexPartial,
                                 ::testing::Values(PartialIndexRow{0, 4, 2, 2}, PartialIndexRow{2, 4, 3, 11},
                                                   PartialIndexRow{3, 1, 0, 3}, PartialIndexRow{1, 0, 0, 0},
                                                   PartialIndexRow{2, 5, 0, 10}));
    } // namespace Test
} // namespace SushiBLAS
