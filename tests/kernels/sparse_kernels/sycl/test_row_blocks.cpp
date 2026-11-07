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
#include <limits>
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
            /** @brief One call of scatter_blocks on a CPU or a GPU profile and the count it must give. */
            struct BlockCountRow
            {
                    const char* name;     /**< The row's suffix. */
                    bool is_cpu;          /**< Whether the profile is a CPU one. */
                    std::size_t units;    /**< The compute units of the profile. */
                    int64_t outputs;      /**< Elements of the result. */
                    int64_t entries;      /**< Stored entries of the matrix. */
                    std::size_t expected; /**< The block count. */
            };

            /**
             * @brief Returns the rows: no outputs, no entries, too few entries, the units bind on a
             *        CPU and on a GPU, the density binds, and the maximum binds on a CPU and on a GPU.
             *
             * A CPU takes its compute units; a GPU takes its compute units times the waves per unit.
             * The density row stores 250 entries for 100 outputs, so 2 * 250 / 100 = 5 partial sums
             * per output are affordable, below the 12 compute units.
             */
            std::vector<BlockCountRow> block_count_rows()
            {
                return {{"NoOutputsTakeNoBlock", true, 8, 0, 100, 0},
                        {"NoEntriesTakeOneBlock", true, 12, 10, 0, 1},
                        {"FewEntriesPerOutputTakeOneBlock", true, 8, 1000, 100, 1},
                        {"TheUnitsCapTheCount", true, 8, 10, 1000000, 8},
                        {"TheUnitsOfAGpuAreTimesTheWaves", false, 4, 10, 1000,
                         4 * Device::K_WAVES_PER_COMPUTE_UNIT},
                        {"TheDensityCapsTheCount", true, 12, 100, 250, 5},
                        {"TheMaximumCapsTheCount", true, 1000, 10, 1000000,
                         Blocks::K_MAX_SCATTER_BLOCKS},
                        {"TheMaximumCapsAGpuCount", false, 64, 10, 100000,
                         Blocks::K_MAX_SCATTER_BLOCKS}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string block_count_row_name(const ::testing::TestParamInfo<BlockCountRow>& info)
            {
                return info.param.name;
            }

            /// @brief Holds the outputs of the fold tests: four outputs, three blocks.
            constexpr int64_t K_FOLD_OUTPUTS = 4;

            /// @brief Holds the blocks of the fold tests.
            constexpr int64_t K_FOLD_BLOCKS = 3;

            /** @brief Returns the partials of three blocks; every output sums to 111, 222, 333, 444. */
            std::vector<double> fold_partials()
            {
                return {1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400};
            }

            /** @brief One fold of output j into y and the value y[j * incy] must end with. */
            struct FoldRow
            {
                    const char* name; /**< The row's suffix. */
                    double alpha;     /**< The scale of the sum. */
                    double beta;      /**< The scale of y. */
                    double y_start;   /**< What every element of y holds before the fold. */
                    int64_t j;        /**< The output folded. */
                    int64_t incy;     /**< The stride of y. */
                    double expected;  /**< The value of y[j * incy] afterwards. */
            };

            /** @brief Returns the rows; the sums of the three blocks are 111, 222, 333 and 444. */
            std::vector<FoldRow> fold_rows()
            {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return {{"ScaledSumPlusScaledY", 2.0, 3.0, 1.0, 0, 1, 225.0},
                        {"BetaZeroIgnoresNanInY", 2.0, 0.0, nan, 1, 1, 444.0},
                        {"BetaOneKeepsY", 1.0, 1.0, 5.0, 3, 1, 449.0},
                        {"StridedOutput", 1.0, 0.0, 7.0, 2, 2, 333.0}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string fold_row_name(const ::testing::TestParamInfo<FoldRow>& info)
            {
                return info.param.name;
            }

            /** @brief Returns y after the host fold of every output, from a y of y_start. */
            std::vector<double> host_fold_all(double alpha, double beta, double y_start)
            {
                const std::vector<double> partials = fold_partials();
                std::vector<double> y(K_FOLD_OUTPUTS, y_start);
                for (int64_t j = 0; j < K_FOLD_OUTPUTS; ++j)
                    Blocks::Detail::fold_block_output<double>(partials.data(), K_FOLD_OUTPUTS,
                                                              K_FOLD_BLOCKS, j, alpha, beta,
                                                              y.data(), 1);
                return y;
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
            Device::Profile profile;
            profile.is_cpu = row.is_cpu;
            profile.compute_units = row.units;

            EXPECT_EQ(Blocks::scatter_blocks(profile, row.outputs, row.entries), row.expected);
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

        /** @brief Folds one output at a time on the host. */
        class Small_RowBlocksFold : public ::testing::TestWithParam<FoldRow>
        {

        };

        TEST_P(Small_RowBlocksFold, AnOutputIsTheScaledSumOfItsSlicesPlusScaledY)
        {
            const FoldRow& row = GetParam();
            const std::vector<double> partials = fold_partials();
            std::vector<double> y(8, row.y_start);

            Blocks::Detail::fold_block_output<double>(partials.data(), K_FOLD_OUTPUTS,
                                                      K_FOLD_BLOCKS, row.j, row.alpha, row.beta,
                                                      y.data(), row.incy);

            EXPECT_EQ(y[static_cast<std::size_t>(row.j * row.incy)], row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_RowBlocksFold, ::testing::ValuesIn(fold_rows()),
                                 fold_row_name);

        TEST(Small_RowBlocksFoldWrites, AnOutputWritesOnlyItsOwnStridedElement)
        {
            const std::vector<double> partials = fold_partials();
            std::vector<double> y(8, 7.0);

            Blocks::Detail::fold_block_output<double>(partials.data(), K_FOLD_OUTPUTS,
                                                      K_FOLD_BLOCKS, 2, 1.0, 0.0, y.data(), 2);

            EXPECT_EQ(y, (std::vector<double>{7, 7, 7, 7, 333, 7, 7, 7}));
        }

        /** @brief Folds partials on the host CPU queue. */
        class Medium_RowBlocksOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_RowBlocksOnDevice, TheLaunchedFoldMatchesTheHostFold)
        {
            const SharedArray<Kernels::Accumulator<double>> partials(queue(), fold_partials());
            SharedArray<double> y(queue(), std::vector<double>(K_FOLD_OUTPUTS, 1.0));

            Blocks::Detail::fold_blocks<double>(queue(), sycl::event(), K_FOLD_OUTPUTS,
                                                K_FOLD_BLOCKS, partials.get(), 2.0, 3.0,
                                                y.get(), 1).wait();

            EXPECT_EQ(y.values(), host_fold_all(2.0, 3.0, 1.0));
        }
    } // namespace Test
} // namespace SushiBLAS
