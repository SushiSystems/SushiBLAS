/**************************************************************************/
/* test_syrk.cpp                                                          */
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
 * @file test_syrk.cpp
 * @brief Tests the reference syrk, single and batched, against a dense rank-k update.
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
#include <SushiBLAS/blas_kernels/reference/level3/syrk.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The order of C.
        constexpr int64_t K_ORDER = 4;

        /// @brief The inner dimension of the update.
        constexpr int64_t K_DEPTH = 3;

        /**
         * @brief Returns C with alpha * G * G^T + beta * C added on one triangle, G n x k.
         * @param g G given row by row.
         */
        static std::vector<double> rank_k(const std::vector<double>& g, std::vector<double> c,
                                   bool upper, double alpha, double beta)
        {
            for (int64_t i = 0; i < K_ORDER; ++i)
            {
                for (int64_t j = 0; j < K_ORDER; ++j)
                {
                    if (!(upper ? (i <= j) : (i >= j)))
                        continue;
                    double sum = 0.0;
                    for (int64_t p = 0; p < K_DEPTH; ++p)
                        sum += element(g, K_DEPTH, i, p) * element(g, K_DEPTH, j, p);
                    double& entry = c[static_cast<std::size_t>(i * K_ORDER + j)];
                    entry = alpha * sum + beta * entry;
                }
            }
            return c;
        }

        /** @brief Returns the transpose of an m x n dense matrix given row by row. */
        static std::vector<double> transpose(const std::vector<double>& dense, int64_t m, int64_t n)
        {
            std::vector<double> out(dense.size());
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j)
                    out[static_cast<std::size_t>(j * m + i)] = element(dense, n, i, j);
            return out;
        }

        /// @brief Names a row after its layout, triangle and transpose.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool, bool>>& info)
        {
            return layout_uplo_label(std::get<0>(info.param), std::get<1>(info.param)) +
                   (std::get<2>(info.param) ? "Trans" : "NoTrans");
        }

        /// @brief Runs syrk for one layout, triangle and transpose, as a row of (row major, upper, transposed).
        class Small_ReferenceSyrk : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

        /// @brief Tests that syrk matches a dense update on the named triangle.
        TEST_P(Small_ReferenceSyrk, MatchesADenseUpdateForEveryLayoutTriangleAndTranspose)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const bool transposed = std::get<2>(GetParam());
            const std::vector<double> g = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 2);
            // op(A) = G, so A is G itself or its transpose.
            const int64_t rows = transposed ? K_DEPTH : K_ORDER;
            const int64_t columns = transposed ? K_ORDER : K_DEPTH;
            const std::vector<double> a_dense = transposed ? transpose(g, K_ORDER, K_DEPTH) : g;
            const std::vector<double> a = in_layout(a_dense, rows, columns, layout);
            const int64_t lda = (layout == Core::Layout::RowMajor) ? columns : rows;
            std::vector<double> stored = in_layout(c, K_ORDER, K_ORDER, layout);

            Kernels::Reference::syrk(layout, uplo_of(upper), transpose_of(transposed), K_ORDER,
                                     K_DEPTH, 2.0, a.data(), lda, -1.0, stored.data(), K_ORDER);

            EXPECT_EQ(stored, in_layout(rank_k(g, c, upper, 2.0, -1.0), K_ORDER, K_ORDER, layout));
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsTrianglesAndTransposes, Small_ReferenceSyrk,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool()),
                                 row_name);

        /// @brief Names a row after the matrix of the batch it checks.
        static std::string matrix_name(const ::testing::TestParamInfo<int64_t>& info)
        {
            return "Matrix" + std::to_string(info.param);
        }

        /// @brief Runs the batched syrk and checks one matrix of the batch, as a row of (matrix).
        class Small_ReferenceSyrkBatch : public ::testing::TestWithParam<int64_t> {};

        /// @brief Tests that the batched syrk updates one matrix as a dense update would and leaves the rest.
        TEST_P(Small_ReferenceSyrkBatch, TheBatchedFormUpdatesEachMatrixOfTheBatch)
        {
            const int64_t matrix = GetParam();
            const int64_t batch = 2;
            const int64_t panel = K_ORDER * K_DEPTH;
            const int64_t square = K_ORDER * K_ORDER;
            const std::vector<double> a = small_integers<double>(batch * panel, 3);
            const std::vector<double> c = small_integers<double>(batch * square, 4);
            const auto slice = [](const std::vector<double>& all, int64_t index, int64_t size)
            {
                return std::vector<double>(all.begin() + index * size,
                                           all.begin() + (index + 1) * size);
            };
            const std::vector<double> updated =
                rank_k(slice(a, matrix, panel), slice(c, matrix, square), false, 1.0, 1.0);
            std::vector<double> expected = c;
            std::copy(updated.begin(), updated.end(), expected.begin() + matrix * square);
            std::vector<double> batched = c;

            Kernels::Reference::syrk(Core::Layout::RowMajor, Core::Uplo::Lower,
                                     Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 1.0, a.data(),
                                     K_DEPTH, panel, 1.0, batched.data(), K_ORDER, square, batch);

            EXPECT_EQ(batched, expected);
        }

        INSTANTIATE_TEST_SUITE_P(Matrices, Small_ReferenceSyrkBatch, ::testing::Values(0, 1),
                                 matrix_name);
    } // namespace Test
} // namespace SushiBLAS
