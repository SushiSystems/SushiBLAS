/**************************************************************************/
/* test_trsm.cpp                                                          */
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
 * @file test_trsm.cpp
 * @brief Tests the blocked SYCL trsm against the reference trsm on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/trsm.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/trsm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The rows of B.
        constexpr int64_t K_ROWS = 37;

        /// @brief The columns of B.
        constexpr int64_t K_COLUMNS = 5;

        /**
         * @brief Returns op(A) * X on the left or X * op(A) on the right, all given row by row.
         * @param a The order x order matrix whose triangle is op's input.
         * @param unit_diagonal Whether the diagonal of the triangle is read as ones, or as stored.
         */
        static std::vector<double> times_op(const std::vector<double>& a, int64_t order,
                                            bool upper, bool transposed, bool left,
                                            const std::vector<double>& x, bool unit_diagonal = true)
        {
            const std::vector<double> t = triangle_of(a, order, upper, unit_diagonal);
            const auto op = [&](int64_t i, int64_t j)
            {
                return transposed ? element(t, order, j, i) : element(t, order, i, j);
            };
            std::vector<double> b(x.size(), 0.0);
            for (int64_t i = 0; i < K_ROWS; ++i)
                for (int64_t j = 0; j < K_COLUMNS; ++j)
                    for (int64_t k = 0; k < order; ++k)
                        b[static_cast<std::size_t>(i * K_COLUMNS + j)] +=
                            left ? op(i, k) * element(x, K_COLUMNS, k, j)
                                 : element(x, K_COLUMNS, i, k) * op(k, j);
            return b;
        }

        /// @brief Returns the matrix with every diagonal entry set to pivot.
        static std::vector<double> with_pivot(std::vector<double> matrix, int64_t order, double pivot)
        {
            for (int64_t i = 0; i < order; ++i)
                matrix[static_cast<std::size_t>(i * order + i)] = pivot;
            return matrix;
        }

        /// @brief Names a row after its block width, side, triangle and transpose.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<int64_t, bool, bool, bool>>& info)
        {
            return "Block" + std::to_string(std::get<0>(info.param)) +
                   (std::get<1>(info.param) ? "Left" : "Right") +
                   (std::get<2>(info.param) ? "Upper" : "Lower") +
                   (std::get<3>(info.param) ? "Trans" : "NoTrans");
        }

        /// @brief Runs trsm on the host CPU queue, one block width, side, triangle and transpose at a time.
        class Medium_SYCLTrsm : public HostDeviceTest, public ::testing::WithParamInterface<std::tuple<int64_t, bool, bool, bool>>
        {

        };

        /// @brief Tests that the SYCL trsm matches the reference for the block width, side, triangle and transpose of a row.
        TEST_P(Medium_SYCLTrsm, MatchesTheReference)
        {
            const int64_t block = std::get<0>(GetParam());
            const bool left = std::get<1>(GetParam());
            const bool upper = std::get<2>(GetParam());
            const bool transposed = std::get<3>(GetParam());
            SharedArray<Kernels::Accumulator<double>> work(queue(),
                                                           static_cast<std::size_t>(K_ROWS *
                                                                                    K_COLUMNS));
            const Kernels::SYCL::TriangularSolvePlan plan{block};
            const int64_t order = left ? K_ROWS : K_COLUMNS;
            // B made from a solution of small integers keeps every step exact.
            const std::vector<double> a = unit_diagonal_matrix<double>(order, 1);
            const SharedArray<double> device_a(queue(), a);
            const std::vector<double> x = small_integers<double>(K_ROWS * K_COLUMNS, 2);
            std::vector<double> b = times_op(a, order, upper, transposed, left, x);
            SharedArray<double> device_b(queue(), b);

            Kernels::SYCL::trsm(queue(), plan, Core::Layout::RowMajor, side_of(left),
                                uplo_of(upper), transpose_of(transposed), Core::Diag::Unit,
                                K_ROWS, K_COLUMNS, 1.0, device_a.get(), order, 0, device_b.get(),
                                K_COLUMNS, 0, 1, work.get(), {}).wait();
            Kernels::Reference::trsm(Core::Layout::RowMajor, side_of(left), uplo_of(upper),
                                     transpose_of(transposed), Core::Diag::Unit, K_ROWS,
                                     K_COLUMNS, 1.0, a.data(), order, b.data(), K_COLUMNS);

            EXPECT_EQ(device_b.values(), b);
        }

        INSTANTIATE_TEST_SUITE_P(BlockWidthsSidesTrianglesAndTransposes, Medium_SYCLTrsm,
                                 ::testing::Combine(::testing::Values(int64_t{1}, int64_t{8},
                                                                      int64_t{64}),
                                                    ::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool()),
                                 row_name);

        /// @brief Runs trsm with alpha other than one on the host CPU queue.
        class Medium_SYCLTrsmAlpha : public HostDeviceTest
        {

        };

        /// @brief Tests that alpha scales the right-hand side of the SYCL trsm as it does the reference one.
        TEST_F(Medium_SYCLTrsmAlpha, AlphaScalesTheRightHandSide)
        {
            SharedArray<Kernels::Accumulator<double>> work(queue(),
                                                           static_cast<std::size_t>(K_ROWS *
                                                                                    K_COLUMNS));
            const Kernels::SYCL::TriangularSolvePlan plan{8};
            const std::vector<double> a = unit_diagonal_matrix<double>(K_ROWS, 1);
            const SharedArray<double> device_a(queue(), a);
            std::vector<double> b = small_integers<double>(K_ROWS * K_COLUMNS, 2);
            SharedArray<double> device_b(queue(), b);

            Kernels::SYCL::trsm(queue(), plan, Core::Layout::RowMajor, Core::Side::Left,
                                Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::Unit,
                                K_ROWS, K_COLUMNS, 2.0, device_a.get(), K_ROWS, 0, device_b.get(),
                                K_COLUMNS, 0, 1, work.get(), {}).wait();
            Kernels::Reference::trsm(Core::Layout::RowMajor, Core::Side::Left, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, Core::Diag::Unit, K_ROWS,
                                     K_COLUMNS, 2.0, a.data(), K_ROWS, b.data(), K_COLUMNS);

            EXPECT_EQ(device_b.values(), b);
        }

        /// @brief Runs trsm with a stored diagonal that is not one on the host CPU queue.
        class Medium_SYCLTrsmNonUnit : public HostDeviceTest
        {

        };

        /// @brief Tests that a non-unit trsm divides each step by its stored pivot, so a solution of small integers comes back whole.
        TEST_F(Medium_SYCLTrsmNonUnit, EachStepDividesByItsPivot)
        {
            SharedArray<Kernels::Accumulator<double>> work(queue(),
                                                           static_cast<std::size_t>(K_ROWS *
                                                                                    K_COLUMNS));
            const Kernels::SYCL::TriangularSolvePlan plan{8};
            // Pivots of 2 divide exactly; B = A X with X of small integers, so the solve must return X.
            const std::vector<double> a = with_pivot(unit_diagonal_matrix<double>(K_ROWS, 1), K_ROWS, 2.0);
            const SharedArray<double> device_a(queue(), a);
            const std::vector<double> x = small_integers<double>(K_ROWS * K_COLUMNS, 2);
            SharedArray<double> device_b(queue(), times_op(a, K_ROWS, true, false, true, x, false));

            Kernels::SYCL::trsm(queue(), plan, Core::Layout::RowMajor, Core::Side::Left,
                                Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::NonUnit,
                                K_ROWS, K_COLUMNS, 1.0, device_a.get(), K_ROWS, 0, device_b.get(),
                                K_COLUMNS, 0, 1, work.get(), {}).wait();

            EXPECT_EQ(device_b.values(), x);
        }
    } // namespace Test
} // namespace SushiBLAS
