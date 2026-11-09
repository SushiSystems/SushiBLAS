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
 * @brief Tests that the reference trsm recovers X from op(A) * X or X * op(A), single and batched.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/trsm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The rows of B.
        constexpr int64_t K_ROWS = 4;

        /// @brief The columns of B.
        constexpr int64_t K_COLUMNS = 3;

        /** @brief Returns the product of two dense matrices given row by row. */
        static std::vector<double> product(const std::vector<double>& p, int64_t rows,
                                           int64_t inner, const std::vector<double>& q,
                                           int64_t columns)
        {
            std::vector<double> out(static_cast<std::size_t>(rows * columns), 0.0);
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t j = 0; j < columns; ++j)
                    for (int64_t k = 0; k < inner; ++k)
                        out[static_cast<std::size_t>(i * columns + j)] +=
                            element(p, inner, i, k) * element(q, columns, k, j);
            return out;
        }

        /** @brief Returns the transpose of an n x n dense matrix. */
        static std::vector<double> transpose(const std::vector<double>& square, int64_t n)
        {
            std::vector<double> out(square.size());
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j)
                    out[static_cast<std::size_t>(j * n + i)] = element(square, n, i, j);
            return out;
        }

        /// @brief Names a row after its layout, side, triangle and transpose.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool, bool, bool>>& info)
        {
            return std::string(std::get<1>(info.param) ? "Left" : "Right") +
                   layout_uplo_label(std::get<0>(info.param), std::get<2>(info.param)) +
                   (std::get<3>(info.param) ? "Trans" : "NoTrans");
        }

        /// @brief Runs trsm for one layout, side, triangle and transpose, as a row of (row major, left, upper, transposed).
        class Small_ReferenceTrsmFlags : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool>> {};

        /// @brief Tests that trsm recovers X from op(A) X or X op(A).
        TEST_P(Small_ReferenceTrsmFlags, RecoversXForEverySideTriangleAndTranspose)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool left = std::get<1>(GetParam());
            const bool upper = std::get<2>(GetParam());
            const bool transposed = std::get<3>(GetParam());
            const std::vector<double> x = small_integers<double>(K_ROWS * K_COLUMNS, 1);
            const int64_t order = left ? K_ROWS : K_COLUMNS;
            // A unit diagonal makes the solve exact in integers.
            const std::vector<double> a = unit_diagonal_matrix<double>(order, 2);
            const std::vector<double> triangle = triangle_of(a, order, upper, false);
            const std::vector<double> op_a = transposed ? transpose(triangle, order) : triangle;
            const std::vector<double> b = left ? product(op_a, K_ROWS, K_ROWS, x, K_COLUMNS)
                                               : product(x, K_ROWS, K_COLUMNS, op_a, K_COLUMNS);
            std::vector<double> stored = in_layout(b, K_ROWS, K_COLUMNS, layout);
            const int64_t ldb = (layout == Core::Layout::RowMajor) ? K_COLUMNS : K_ROWS;

            Kernels::Reference::trsm(layout, side_of(left), uplo_of(upper), transpose_of(transposed),
                                     Core::Diag::NonUnit, K_ROWS, K_COLUMNS, 1.0,
                                     in_layout(a, order, order, layout).data(), order,
                                     stored.data(), ldb);

            EXPECT_EQ(stored, in_layout(x, K_ROWS, K_COLUMNS, layout));
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsSidesTrianglesAndTransposes, Small_ReferenceTrsmFlags,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool(), ::testing::Bool()),
                                 row_name);

        /// @brief Tests that alpha scales the right-hand side before the solve.
        TEST(Small_ReferenceTrsm, AlphaScalesTheRightHandSide)
        {
            const std::vector<double> a{1.0, 0.0, 2.0, 1.0};
            std::vector<double> b{2.0, 6.0};

            Kernels::Reference::trsm(Core::Layout::RowMajor, Core::Side::Left, Core::Uplo::Lower,
                                     Core::Transpose::NoTrans, Core::Diag::Unit, 2, 1, 0.5,
                                     a.data(), 2, b.data(), 1);

            EXPECT_EQ(b, (std::vector<double>{1.0, 1.0}));
        }

        /// @brief Names a row after the system of the batch it checks.
        static std::string system_name(const ::testing::TestParamInfo<int64_t>& info)
        {
            return "System" + std::to_string(info.param);
        }

        /// @brief Runs the batched trsm and checks one system of the batch, as a row of (system).
        class Small_ReferenceTrsmBatch : public ::testing::TestWithParam<int64_t> {};

        /// @brief Tests that the batched trsm solves one system as a single call would and leaves the rest.
        TEST_P(Small_ReferenceTrsmBatch, TheBatchedFormSolvesEachSystemOfTheBatch)
        {
            const int64_t system = GetParam();
            const int64_t batch = 2;
            const int64_t size = K_ROWS * K_COLUMNS;
            // A is shared by stride zero, upper with a unit diagonal, so the solve is exact.
            const std::vector<double> a = unit_diagonal_matrix<double>(K_ROWS, 3);
            const std::vector<double> x = small_integers<double>(size, 4);
            const std::vector<double> product_b =
                product(triangle_of(a, K_ROWS, true, false), K_ROWS, K_ROWS, x, K_COLUMNS);
            // Only the checked system holds a right-hand side; the others are zero systems.
            std::vector<double> batched(static_cast<std::size_t>(batch * size), 0.0);
            std::vector<double> expected = batched;
            std::copy(product_b.begin(), product_b.end(), batched.begin() + system * size);
            std::copy(x.begin(), x.end(), expected.begin() + system * size);

            Kernels::Reference::trsm(Core::Layout::RowMajor, Core::Side::Left, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, Core::Diag::Unit, K_ROWS,
                                     K_COLUMNS, 1.0, a.data(), K_ROWS, 0, batched.data(),
                                     K_COLUMNS, size, batch);

            EXPECT_EQ(batched, expected);
        }

        INSTANTIATE_TEST_SUITE_P(Systems, Small_ReferenceTrsmBatch, ::testing::Values(0, 1),
                                 system_name);
    } // namespace Test
} // namespace SushiBLAS
