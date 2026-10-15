/**************************************************************************/
/* test_row_reduction.cpp                                                 */
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
 * @file test_row_reduction.cpp
 * @brief Tests the row width a profile gives and the group fold of whole rows and triangle rows.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/row_reduction.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Rows = Kernels::SYCL::Detail;

        /// @brief The rows of the matrix the folds run over.
        constexpr int64_t K_ROWS = 6;

        /// @brief The columns of that matrix.
        constexpr int64_t K_COLUMNS = 100;

        /// @brief The group width the folds run at; a power of two, as the kernel requires.
        constexpr std::size_t K_WIDTH = 16;

        TEST(Unit_RowReduction, ARowShorterThanTheStagedThresholdIsNotSplit)
        {
            Device::Profile cpu;
            cpu.is_cpu = true;
            cpu.compute_units = 8;
            cpu.max_work_group_size = 64;

            EXPECT_EQ(Rows::row_reduction_width(cpu, 0), std::size_t{0});
            EXPECT_EQ(Rows::row_reduction_width(cpu, 100), std::size_t{0});
            EXPECT_EQ(Rows::row_reduction_width(cpu, std::int64_t{1} << 20),
                      Reduce::group_width(cpu));
        }

        TEST(Unit_RowReduction, TheWindowsCoverAWholeRowOrOneTriangle)
        {
            const Rows::RowWindow whole = Rows::WholeRow{K_COLUMNS}(3);
            const Rows::RowWindow upper = Rows::TriangleRow{10, true}(3);
            const Rows::RowWindow lower = Rows::TriangleRow{10, false}(3);

            EXPECT_EQ(whole.first, int64_t{0});
            EXPECT_EQ(whole.last, K_COLUMNS);
            EXPECT_EQ(upper.first, int64_t{3});
            EXPECT_EQ(upper.last, int64_t{10});
            EXPECT_EQ(lower.first, int64_t{0});
            EXPECT_EQ(lower.last, int64_t{4});
        }

        /** @brief Runs the group folds on the host CPU queue. */
        class Device_RowReduction : public HostDeviceTest
        {

        };

        TEST_F(Device_RowReduction, AGroupFoldsEveryRowToItsSum)
        {
            const std::vector<double> values = small_integers<double>(K_ROWS * K_COLUMNS, 1);
            const SharedArray<double> matrix(queue(), values);
            SharedArray<double> sums(queue(), static_cast<std::size_t>(K_ROWS));
            const double* m = matrix.get();
            double* out = sums.get();

            queue().submit([&](sycl::handler& h)
            {
                Rows::reduce_rows_by_group<double>(
                    h, K_ROWS, K_COLUMNS, K_WIDTH,
                    [=](int64_t row, int64_t k) { return m[row * K_COLUMNS + k]; },
                    [=](int64_t row, double sum) { out[row] = sum; });
            }).wait();

            for (int64_t row = 0; row < K_ROWS; ++row)
            {
                double expected = 0.0;
                for (int64_t k = 0; k < K_COLUMNS; ++k)
                    expected += values[static_cast<std::size_t>(row * K_COLUMNS + k)];
                EXPECT_EQ(sums.get()[row], expected) << "row " << row;
            }
        }

        TEST_F(Device_RowReduction, ATriangleWindowFoldsOnlyItsColumns)
        {
            const int64_t order = K_ROWS;
            SharedArray<double> sums(queue(), static_cast<std::size_t>(order));
            double* out = sums.get();

            queue().submit([&](sycl::handler& h)
            {
                Rows::reduce_row_windows_by_group<double>(
                    h, order, K_WIDTH, Rows::TriangleRow{order, true},
                    [](int64_t, int64_t) { return 1.0; },
                    [=](int64_t row, double sum) { out[row] = sum; });
            }).wait();

            for (int64_t row = 0; row < order; ++row)
                EXPECT_EQ(sums.get()[row], static_cast<double>(order - row)) << "row " << row;
        }
    } // namespace Test
} // namespace SushiBLAS
