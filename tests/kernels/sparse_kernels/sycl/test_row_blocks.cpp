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
 * @file test_row_blocks.cpp
 * @brief Tests how many row blocks a scattering product takes, where each starts, and their fold.
 * @author Mustafa Garip
 */

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

        TEST(Medium_RowBlocks, TheBlockCountIsCappedByUnitsEntriesAndTheMaximum)
        {
            Device::Profile cpu;
            cpu.is_cpu = true;
            cpu.compute_units = 8;

            EXPECT_EQ(Blocks::scatter_blocks(cpu, 0, 100), std::size_t{0});
            EXPECT_EQ(Blocks::scatter_blocks(cpu, 1000, 100), std::size_t{1});
            EXPECT_EQ(Blocks::scatter_blocks(cpu, 10, 1000000), std::size_t{8});
            EXPECT_LE(Blocks::scatter_blocks(cpu, 10, 1000000), Blocks::K_MAX_SCATTER_BLOCKS);
        }

        TEST(Medium_RowBlocks, TheBlocksCoverTheRowsInOrder)
        {
            const IntegerCsr a = integer_csr(50, 40, 3, false);
            const Kernels::CsrPattern<int64_t> pattern{50, 40, a.nnz(), 0, a.offsets.data(),
                                                       a.columns.data()};
            const int64_t blocks = 6;

            EXPECT_EQ(Blocks::Detail::block_first_row(pattern, 0, blocks), int64_t{0});
            EXPECT_EQ(Blocks::Detail::block_end_row(pattern, blocks - 1, blocks), int64_t{50});
            for (int64_t block = 0; block + 1 < blocks; ++block)
                EXPECT_EQ(Blocks::Detail::block_end_row(pattern, block, blocks),
                          Blocks::Detail::block_first_row(pattern, block + 1, blocks));
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
