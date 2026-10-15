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
 * @file test_trsm.cpp
 * @brief Tests the blocked SYCL trsm against the reference trsm on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
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
         * @param a The order x order matrix whose triangle, with a unit diagonal, is op's input.
         */
        static std::vector<double> times_op(const std::vector<double>& a, int64_t order,
                                            bool upper, bool transposed, bool left,
                                            const std::vector<double>& x)
        {
            const std::vector<double> t = triangle_of(a, order, upper, true);
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

        /** @brief Runs trsm on the host CPU queue. */
        class Medium_SYCLTrsm : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLTrsm, MatchesTheReferenceForEveryFlagAndBlockWidth)
        {
            SharedArray<Kernels::Accumulator<double>> work(queue(),
                                                           static_cast<std::size_t>(K_ROWS *
                                                                                    K_COLUMNS));
            for (const int64_t block : {int64_t{1}, int64_t{8}, int64_t{64}})
            {
                const Kernels::SYCL::TriangularSolvePlan plan{block};
                for (const bool left : {true, false})
                {
                    const int64_t order = left ? K_ROWS : K_COLUMNS;
                    // B made from a solution of small integers keeps every step exact.
                    const std::vector<double> a = unit_diagonal_matrix<double>(order, 1);
                    const SharedArray<double> device_a(queue(), a);
                    for (const bool upper : {false, true})
                    {
                        for (const bool transposed : {false, true})
                        {
                            SCOPED_TRACE(100 * block + 4 * left + 2 * upper + transposed);
                            const std::vector<double> x =
                                small_integers<double>(K_ROWS * K_COLUMNS, 2);
                            std::vector<double> b =
                                times_op(a, order, upper, transposed, left, x);
                            SharedArray<double> device_b(queue(), b);

                            Kernels::SYCL::trsm(queue(), plan, Core::Layout::RowMajor,
                                                side_of(left), uplo_of(upper),
                                                transpose_of(transposed), Core::Diag::Unit,
                                                K_ROWS, K_COLUMNS, 1.0, device_a.get(), order,
                                                0, device_b.get(), K_COLUMNS, 0, 1, work.get(),
                                                {}).wait();
                            Kernels::Reference::trsm(Core::Layout::RowMajor, side_of(left),
                                                     uplo_of(upper), transpose_of(transposed),
                                                     Core::Diag::Unit, K_ROWS, K_COLUMNS, 1.0,
                                                     a.data(), order, b.data(), K_COLUMNS);

                            EXPECT_EQ(device_b.values(), b);
                            EXPECT_EQ(b, x);
                        }
                    }
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
