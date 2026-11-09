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
 * @file test_row_reduction.cpp
 * @brief Tests that gemv and trmv fold short rows, long rows and triangle rows to their sums, in a fixed order.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/gemv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/trmv.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The rows of the matrix the folds run over.
            constexpr int64_t K_ROWS = 6;

            /// @brief Names a row after its length.
            std::string length_name(const ::testing::TestParamInfo<int64_t>& info)
            {
                return "Length" + std::to_string(info.param);
            }

            /** @brief Returns the sum of every row of a matrix of the given rows, given row by row. */
            [[nodiscard]] std::vector<double> row_sums(const std::vector<double>& values,
                                                       int64_t rows, int64_t columns)
            {
                std::vector<double> sums(static_cast<std::size_t>(rows), 0.0);
                for (int64_t row = 0; row < rows; ++row)
                    for (int64_t k = 0; k < columns; ++k)
                        sums[static_cast<std::size_t>(row)] +=
                            values[static_cast<std::size_t>(row * columns + k)];
                return sums;
            }

            /** @brief Returns the counts n, n - 1, ... 1 of the columns each upper triangle row holds. */
            [[nodiscard]] std::vector<double> triangle_counts(int64_t order)
            {
                std::vector<double> counts(static_cast<std::size_t>(order));
                for (int64_t row = 0; row < order; ++row)
                    counts[static_cast<std::size_t>(row)] = static_cast<double>(order - row);
                return counts;
            }

            /** @brief Returns terms of both signs and of magnitudes 1e7 and below 1.3, in which every order of the sum gives other bits. */
            [[nodiscard]] std::vector<float> ill_conditioned(std::size_t count)
            {
                std::vector<float> values(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const float magnitude =
                        (i % 7 == 0) ? 1.0e7f : 0.1f * static_cast<float>(i % 13);
                    values[i] = (i % 2 == 0) ? magnitude : -magnitude;
                }
                return values;
            }
        } // namespace

        /// @brief Runs gemv, which folds each row of a long matrix by a group, on the host CPU queue, one row length at a time.
        class Medium_RowReductionOnDevice : public HostDeviceTest,
                                            public ::testing::WithParamInterface<int64_t>
        {

        };

        /// @brief Tests that gemv folds every row of a matrix to its sum, short rows and long rows alike.
        TEST_P(Medium_RowReductionOnDevice, AGroupFoldsEveryRowToItsSum)
        {
            const int64_t columns = GetParam();
            // The elements are small integers and x is ones, so every order of the sum is exact.
            const std::vector<double> values = small_integers<double>(K_ROWS * columns, 1);
            const std::vector<double> ones(static_cast<std::size_t>(columns), 1.0);
            const SharedArray<double> matrix(queue(), values);
            const SharedArray<double> x(queue(), ones);
            SharedArray<double> sums(queue(), static_cast<std::size_t>(K_ROWS));

            Kernels::SYCL::gemv(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans, K_ROWS,
                                columns, 1.0, matrix.get(), columns, x.get(), 1, 0.0, sums.get(), 1,
                                {}).wait();

            EXPECT_EQ(sums.values(), row_sums(values, K_ROWS, columns));
        }

        INSTANTIATE_TEST_SUITE_P(Lengths, Medium_RowReductionOnDevice,
                                 ::testing::Values(int64_t{100}, int64_t{5000}, int64_t{70000}),
                                 length_name);

        /// @brief Runs gemv twice on the host CPU queue, one row length at a time.
        class Medium_RowReductionOrder : public HostDeviceTest,
                                         public ::testing::WithParamInterface<int64_t>
        {

        };

        /// @brief Tests that a fold adds in a fixed order, so two runs on terms that rounding makes order dependent give the same bits.
        TEST_P(Medium_RowReductionOrder, AGroupFoldGivesTheSameBitsOnEveryRun)
        {
            const int64_t columns = GetParam();
            const std::vector<float> values =
                ill_conditioned(static_cast<std::size_t>(K_ROWS * columns));
            const std::vector<float> ones(static_cast<std::size_t>(columns), 1.0f);
            const SharedArray<float> matrix(queue(), values);
            const SharedArray<float> x(queue(), ones);
            SharedArray<float> first(queue(), static_cast<std::size_t>(K_ROWS));
            SharedArray<float> second(queue(), static_cast<std::size_t>(K_ROWS));

            Kernels::SYCL::gemv(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans, K_ROWS,
                                columns, 1.0f, matrix.get(), columns, x.get(), 1, 0.0f,
                                first.get(), 1, {}).wait();
            Kernels::SYCL::gemv(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans, K_ROWS,
                                columns, 1.0f, matrix.get(), columns, x.get(), 1, 0.0f,
                                second.get(), 1, {}).wait();

            EXPECT_EQ(first.values(), second.values());
        }

        INSTANTIATE_TEST_SUITE_P(Lengths, Medium_RowReductionOrder,
                                 ::testing::Values(int64_t{100}, int64_t{70000}), length_name);

        /// @brief Runs trmv, which folds the triangle window of each row by a group, on the host CPU queue, one order at a time.
        class Medium_RowReductionTriangle : public HostDeviceTest,
                                            public ::testing::WithParamInterface<int64_t>
        {

        };

        /// @brief Tests that trmv folds only the columns of the upper triangle of each row.
        TEST_P(Medium_RowReductionTriangle, ATriangleWindowFoldsOnlyItsColumns)
        {
            const int64_t order = GetParam();
            // An upper triangle of ones times a vector of ones leaves order - row in each row.
            const std::vector<double> ones_matrix(static_cast<std::size_t>(order * order), 1.0);
            const std::vector<double> ones(static_cast<std::size_t>(order), 1.0);
            const SharedArray<double> matrix(queue(), ones_matrix);
            SharedArray<double> x(queue(), ones);
            SharedArray<double> work(queue(), static_cast<std::size_t>(order));

            Kernels::SYCL::trmv(queue(), Core::Layout::RowMajor, Core::Uplo::Upper,
                                Core::Transpose::NoTrans, Core::Diag::NonUnit, order, matrix.get(),
                                order, x.get(), 1, work.get(), {}).wait();

            EXPECT_EQ(x.values(), triangle_counts(order));
        }

        INSTANTIATE_TEST_SUITE_P(Orders, Medium_RowReductionTriangle,
                                 ::testing::Values(int64_t{6}, int64_t{300}), length_name);
    } // namespace Test
} // namespace SushiBLAS
