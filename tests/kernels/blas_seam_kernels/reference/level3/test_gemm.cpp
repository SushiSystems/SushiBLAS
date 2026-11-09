/**************************************************************************/
/* test_gemm.cpp                                                          */
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
 * @file test_gemm.cpp
 * @brief Tests the reference gemm against a dense loop, with transposes, batches and an epilogue.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Rows of C.
        constexpr int64_t K_M = 4;

        /// @brief Columns of C.
        constexpr int64_t K_N = 5;

        /// @brief Terms of each sum.
        constexpr int64_t K_K = 3;

        /** @brief Returns the transpose of a rows x columns matrix given row by row. */
        static std::vector<double> transposed_of(const std::vector<double>& m, int64_t rows,
                                                 int64_t columns)
        {
            std::vector<double> out(m.size());
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t j = 0; j < columns; ++j)
                    out[static_cast<std::size_t>(j * rows + i)] = element(m, columns, i, j);
            return out;
        }

        /** @brief Returns alpha * A * B + beta * C for A m x k, B k x n, all row by row. */
        static std::vector<double> dense_gemm(const std::vector<double>& a,
                                              const std::vector<double>& b, double alpha,
                                              double beta, std::vector<double> c)
        {
            for (int64_t i = 0; i < K_M; ++i)
            {
                for (int64_t j = 0; j < K_N; ++j)
                {
                    double sum = 0.0;
                    for (int64_t p = 0; p < K_K; ++p)
                        sum += element(a, K_K, i, p) * element(b, K_N, p, j);
                    double& entry = c[static_cast<std::size_t>(i * K_N + j)];
                    entry = alpha * sum + beta * entry;
                }
            }
            return c;
        }

        /** @brief Holds one layout and the two transposes of a product. */
        struct GemmCase
        {
            Core::Layout layout; /**< The storage order of A, B and C. */
            bool trans_a;        /**< Whether A is transposed. */
            bool trans_b;        /**< Whether B is transposed. */
        };

        /** @brief Returns both layouts crossed with the four transpose pairs. */
        static std::vector<GemmCase> gemm_cases()
        {
            std::vector<GemmCase> cases;
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
                for (const bool trans_a : {false, true})
                    for (const bool trans_b : {false, true})
                        cases.push_back({layout, trans_a, trans_b});
            return cases;
        }

        /** @brief Names a case for the test name. */
        static std::string gemm_case_name(const ::testing::TestParamInfo<GemmCase>& info)
        {
            return std::string(info.param.layout == Core::Layout::RowMajor ? "RowMajor"
                                                                            : "ColumnMajor") +
                   (info.param.trans_a ? "TransA" : "PlainA") +
                   (info.param.trans_b ? "TransB" : "PlainB");
        }

        /** @brief Returns the dense product with the bias of every column added to its elements. */
        static std::vector<double> biased(std::vector<double> product,
                                          const std::vector<double>& bias)
        {
            for (int64_t i = 0; i < K_M; ++i)
                for (int64_t j = 0; j < K_N; ++j)
                    product[static_cast<std::size_t>(i * K_N + j)] +=
                        bias[static_cast<std::size_t>(j)];
            return product;
        }

        /** @brief Returns the batch of products that single gemm calls give, one per matrix. */
        static std::vector<double> single_calls(const std::vector<double>& a,
                                                const std::vector<double>& b, int64_t batch)
        {
            std::vector<double> single(static_cast<std::size_t>(batch * K_M * K_N), 1.0);
            for (int64_t i = 0; i < batch; ++i)
                Kernels::Reference::gemm(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                         Core::Transpose::NoTrans, K_M, K_N, K_K, 1.0,
                                         a.data() + i * K_M * K_K, K_K,
                                         b.data() + i * K_K * K_N, K_N, 1.0,
                                         single.data() + i * K_M * K_N, K_N);
            return single;
        }

        /** @brief Runs the reference gemm in one layout and transpose pair at a time. */
        class Small_ReferenceGemm : public ::testing::TestWithParam<GemmCase>
        {

        };

        TEST_P(Small_ReferenceGemm, MatchesADenseLoopInEveryLayoutAndTranspose)
        {
            const GemmCase row = GetParam();
            const std::vector<double> a = small_integers<double>(K_M * K_K, 1);
            const std::vector<double> b = small_integers<double>(K_K * K_N, 2);
            const std::vector<double> c = small_integers<double>(K_M * K_N, 3);
            const std::vector<double> expected = dense_gemm(a, b, 2.0, -1.0, c);
            const int64_t a_rows = row.trans_a ? K_K : K_M;
            const int64_t a_cols = row.trans_a ? K_M : K_K;
            const int64_t b_rows = row.trans_b ? K_N : K_K;
            const int64_t b_cols = row.trans_b ? K_K : K_N;
            const std::vector<double> stored_a = in_layout(
                row.trans_a ? transposed_of(a, K_M, K_K) : a, a_rows, a_cols, row.layout);
            const std::vector<double> stored_b = in_layout(
                row.trans_b ? transposed_of(b, K_K, K_N) : b, b_rows, b_cols, row.layout);
            const bool row_major = (row.layout == Core::Layout::RowMajor);
            std::vector<double> got = in_layout(c, K_M, K_N, row.layout);

            Kernels::Reference::gemm(row.layout, transpose_of(row.trans_a),
                                     transpose_of(row.trans_b), K_M, K_N, K_K, 2.0,
                                     stored_a.data(), row_major ? a_cols : a_rows,
                                     stored_b.data(), row_major ? b_cols : b_rows, -1.0,
                                     got.data(), row_major ? K_N : K_M);

            EXPECT_EQ(got, in_layout(expected, K_M, K_N, row.layout));
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceGemm, ::testing::ValuesIn(gemm_cases()),
                                 gemm_case_name);

        TEST(Small_ReferenceGemmForms, AnEpilogueSeesEachElementAfterAlphaAndBeta)
        {
            const std::vector<double> a = small_integers<double>(K_M * K_K, 4);
            const std::vector<double> b = small_integers<double>(K_K * K_N, 5);
            const std::vector<double> bias = small_integers<double>(K_N, 6);
            const std::vector<double> expected = biased(
                dense_gemm(a, b, 1.0, 0.0, std::vector<double>(K_M * K_N, 0.0)), bias);
            std::vector<double> got(K_M * K_N, 0.0);

            Kernels::Reference::gemm(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                     Core::Transpose::NoTrans, K_M, K_N, K_K, 1.0, a.data(), K_K,
                                     b.data(), K_N, 0.0, got.data(), K_N,
                                     Kernels::ChannelBiasEpilogue{bias.data(), 0, 1});

            EXPECT_EQ(got, expected);
        }

        TEST(Small_ReferenceGemmForms, TheBatchedFormAgreesWithTheSingleCall)
        {
            const int64_t batch = 2;
            const std::vector<double> a = small_integers<double>(batch * K_M * K_K, 7);
            const std::vector<double> b = small_integers<double>(batch * K_K * K_N, 8);
            std::vector<double> batched(batch * K_M * K_N, 1.0);

            Kernels::Reference::gemm(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                     Core::Transpose::NoTrans, K_M, K_N, K_K, 1.0, a.data(), K_K,
                                     K_M * K_K, b.data(), K_N, K_K * K_N, 1.0, batched.data(),
                                     K_N, K_M * K_N, batch);

            EXPECT_EQ(batched, single_calls(a, b, batch));
        }

        TEST(Small_ReferenceGemmForms, TheAccessedFormAgreesWithTheSingleCall)
        {
            const int64_t batch = 2;
            const std::vector<double> a = small_integers<double>(batch * K_M * K_K, 7);
            const std::vector<double> b = small_integers<double>(batch * K_K * K_N, 8);
            std::vector<double> accessed(batch * K_M * K_N, 1.0);

            Kernels::Reference::gemm_accessed(
                K_M, K_N, K_K, 1.0, a.data(),
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, K_K),
                K_M * K_K, b.data(),
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, K_N),
                K_K * K_N, 1.0, accessed.data(),
                Kernels::operand_access(Core::Layout::RowMajor, Core::Transpose::NoTrans, K_N),
                K_M * K_N, batch);

            EXPECT_EQ(accessed, single_calls(a, b, batch));
        }
    } // namespace Test
} // namespace SushiBLAS
