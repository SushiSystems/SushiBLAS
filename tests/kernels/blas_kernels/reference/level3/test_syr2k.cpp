/**************************************************************************/
/* test_syr2k.cpp                                                         */
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
 * @file test_syr2k.cpp
 * @brief Tests the reference syr2k, single and batched, against a dense rank-2k update.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syr2k.hpp>
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
         * @brief Returns C with alpha * (G * H^T + H * G^T) + beta * C on one triangle, G and H n x k.
         * @param g G given row by row.
         * @param h H given row by row.
         */
        static std::vector<double> rank_two_k(const std::vector<double>& g,
                                              const std::vector<double>& h,
                                              std::vector<double> c, bool upper, double alpha,
                                              double beta)
        {
            for (int64_t i = 0; i < K_ORDER; ++i)
            {
                for (int64_t j = 0; j < K_ORDER; ++j)
                {
                    if (!(upper ? (i <= j) : (i >= j)))
                        continue;
                    double sum = 0.0;
                    for (int64_t p = 0; p < K_DEPTH; ++p)
                        sum += element(g, K_DEPTH, i, p) * element(h, K_DEPTH, j, p) +
                               element(h, K_DEPTH, i, p) * element(g, K_DEPTH, j, p);
                    double& entry = c[static_cast<std::size_t>(i * K_ORDER + j)];
                    entry = alpha * sum + beta * entry;
                }
            }
            return c;
        }

        /// @brief Names a row after its triangle.
        static std::string triangle_name(const ::testing::TestParamInfo<bool>& info)
        {
            return info.param ? "Upper" : "Lower";
        }

        /// @brief Runs syr2k for one triangle, as a row of (upper).
        class Small_ReferenceSyr2k : public ::testing::TestWithParam<bool> {};

        /// @brief Tests that syr2k matches a dense update on the named triangle.
        TEST_P(Small_ReferenceSyr2k, MatchesADenseUpdateOnEachTriangle)
        {
            const bool upper = GetParam();
            // A and B are n x k, row-major and not transposed.
            const std::vector<double> a = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const std::vector<double> b = small_integers<double>(K_ORDER * K_DEPTH, 2);
            const std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 3);
            std::vector<double> got = c;

            Kernels::Reference::syr2k(Core::Layout::RowMajor, uplo_of(upper),
                                      Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 3.0,
                                      a.data(), K_DEPTH, b.data(), K_DEPTH, 2.0, got.data(),
                                      K_ORDER);

            EXPECT_EQ(got, rank_two_k(a, b, c, upper, 3.0, 2.0));
        }

        INSTANTIATE_TEST_SUITE_P(Triangles, Small_ReferenceSyr2k, ::testing::Bool(), triangle_name);

        /// @brief Names a row after the matrix of the batch it checks.
        static std::string matrix_name(const ::testing::TestParamInfo<int64_t>& info)
        {
            return "Matrix" + std::to_string(info.param);
        }

        /// @brief Runs the batched syr2k and checks one matrix of the batch, as a row of (matrix).
        class Small_ReferenceSyr2kBatch : public ::testing::TestWithParam<int64_t> {};

        /// @brief Tests that the batched syr2k updates each matrix as a dense update of it would.
        TEST_P(Small_ReferenceSyr2kBatch, TheBatchedFormUpdatesEachMatrixOfTheBatch)
        {
            const int64_t matrix = GetParam();
            const int64_t batch = 3;
            const int64_t panel = K_ORDER * K_DEPTH;
            const int64_t square = K_ORDER * K_ORDER;
            const std::vector<double> a = small_integers<double>(batch * panel, 4);
            const std::vector<double> b = small_integers<double>(batch * panel, 5);
            const std::vector<double> c = small_integers<double>(batch * square, 6);
            const auto slice = [](const std::vector<double>& all, int64_t index, int64_t size)
            {
                return std::vector<double>(all.begin() + index * size,
                                           all.begin() + (index + 1) * size);
            };
            // The matrices are stored column-major, so the dense C is the transposed storage.
            const std::vector<double> dense_c = in_layout(slice(c, matrix, square), K_ORDER, K_ORDER,
                                                          Core::Layout::ColumnMajor);
            const std::vector<double> updated = in_layout(
                rank_two_k(slice(a, matrix, panel), slice(b, matrix, panel), dense_c, true, 1.0, 1.0),
                K_ORDER, K_ORDER, Core::Layout::ColumnMajor);
            std::vector<double> batched = c;

            Kernels::Reference::syr2k(Core::Layout::ColumnMajor, Core::Uplo::Upper,
                                      Core::Transpose::Trans, K_ORDER, K_DEPTH, 1.0, a.data(),
                                      K_DEPTH, panel, b.data(), K_DEPTH, panel, 1.0,
                                      batched.data(), K_ORDER, square, batch);

            EXPECT_EQ(slice(batched, matrix, square), updated);
        }

        INSTANTIATE_TEST_SUITE_P(Matrices, Small_ReferenceSyr2kBatch, ::testing::Values(0, 1, 2),
                                 matrix_name);
    } // namespace Test
} // namespace SushiBLAS
