/**************************************************************************/
/* test_row_blocks.cpp                                                    */
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
 * @file test_row_blocks.cpp
 * @brief Tests how many row blocks a scattering product takes, where each starts, and their fold.
 * @author Mustafa Garip
 */

#include <string>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/sycl/row_blocks.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Blocks = Kernels::SYCL;

        namespace
        {
            /** @brief One call of scatter_blocks on a CPU profile and the count it must give. */
            struct BlockCountRow
            {
                    const char* name;     /**< The row's suffix. */
                    std::size_t units;    /**< The compute units of the CPU profile. */
                    int64_t outputs;      /**< Elements of the result. */
                    int64_t entries;      /**< Stored entries of the matrix. */
                    std::size_t expected; /**< The block count. */
            };

            /** @brief Returns the rows: no outputs, too few entries, units bind, and the maximum binds. */
            std::vector<BlockCountRow> block_count_rows()
            {
                return {{"NoOutputsTakeNoBlock", 8, 0, 100, 0},
                        {"FewEntriesPerOutputTakeOneBlock", 8, 1000, 100, 1},
                        {"TheUnitsCapTheCount", 8, 10, 1000000, 8},
                        {"TheMaximumCapsTheCount", 1000, 10, 1000000, Blocks::K_MAX_SCATTER_BLOCKS}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string block_count_row_name(const ::testing::TestParamInfo<BlockCountRow>& info)
            {
                return info.param.name;
            }

            /** @brief Expects the blocks of a matrix to start at row 0, end at the last row, and touch end to start. */
            void expect_blocks_cover_the_rows(const Kernels::CsrPattern<int64_t>& pattern,
                                              int64_t blocks)
            {
                EXPECT_EQ(Blocks::Detail::block_first_row(pattern, 0, blocks), int64_t{0});
                EXPECT_EQ(Blocks::Detail::block_end_row(pattern, blocks - 1, blocks), pattern.rows);
                for (int64_t block = 0; block + 1 < blocks; ++block)
                    EXPECT_EQ(Blocks::Detail::block_end_row(pattern, block, blocks),
                              Blocks::Detail::block_first_row(pattern, block + 1, blocks));
            }
        } // namespace

        /** @brief Runs scatter_blocks one profile at a time. */
        class Small_RowBlocks : public ::testing::TestWithParam<BlockCountRow>
        {

        };

        TEST_P(Small_RowBlocks, TheBlockCountIsCappedByUnitsEntriesAndTheMaximum)
        {
            const BlockCountRow& row = GetParam();
            Device::Profile cpu;
            cpu.is_cpu = true;
            cpu.compute_units = row.units;

            EXPECT_EQ(Blocks::scatter_blocks(cpu, row.outputs, row.entries), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RowBlocks, ::testing::ValuesIn(block_count_rows()),
                                 block_count_row_name);

        TEST(Small_RowBlocksCoverage, TheBlocksCoverTheRowsInOrder)
        {
            const IntegerCsr a = integer_csr(50, 40, 3, false);
            const Kernels::CsrPattern<int64_t> pattern{50, 40, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};

            expect_blocks_cover_the_rows(pattern, 6);
        }

        /** @brief Folds partials on the host CPU queue. */
        class Medium_RowBlocksOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_RowBlocksOnDevice, TheFoldAddsEverySliceThenScales)
        {
            const int64_t outputs = 4;
            const int64_t blocks = 3;
            const SharedArray<Kernels::Accumulator<double>> partials(
                queue(), std::vector<double>{1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400});
            SharedArray<double> y(queue(), std::vector<double>{1, 1, 1, 1});

            Blocks::Detail::fold_blocks<double>(queue(), sycl::event(), outputs, blocks,
                                                partials.get(), 2.0, 3.0, y.get(), 1).wait();

            EXPECT_EQ(y.values(), (std::vector<double>{225.0, 447.0, 669.0, 891.0}));
        }
    } // namespace Test
} // namespace SushiBLAS
