/**************************************************************************/
/* test_gemm_reference.cpp                                                */
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
 * @file test_gemm_reference.cpp
 * @brief Tests the reference gemm kernel on host buffers.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Core::Layout;
        using Core::Transpose;

        /**
         * @brief Tests that a transpose equals a flipped layout, on integer data the host sums exactly.
         *
         * A k x m row-major buffer read transposed is the m x k column-major buffer of the same
         * bytes read as given, so the two calls must give C and its column-major copy.
         */
        TEST(Small_GemmStrideTable, ATransposeEqualsAFlippedLayout)
        {
            constexpr int64_t K_M = 5;
            constexpr int64_t K_N = 4;
            constexpr int64_t K_K = 3;

            std::vector<float> a(static_cast<std::size_t>(K_K * K_M));
            std::vector<float> b(static_cast<std::size_t>(K_K * K_N));
            for (std::size_t i = 0; i < a.size(); ++i)
                a[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
            for (std::size_t i = 0; i < b.size(); ++i)
                b[i] = static_cast<float>(static_cast<int>(i % 5) - 2);

            std::vector<float> row_major_c(static_cast<std::size_t>(K_M * K_N), 0.0f);
            std::vector<float> column_major_c(row_major_c.size(), 0.0f);
            Kernels::Reference::gemm<float>(Layout::RowMajor, Transpose::Trans, Transpose::NoTrans,
                                            K_M, K_N, K_K, 1.0f, a.data(), K_M, b.data(), K_N, 0.0f,
                                            row_major_c.data(), K_N);
            Kernels::Reference::gemm<float>(Layout::ColumnMajor, Transpose::NoTrans, Transpose::Trans,
                                            K_M, K_N, K_K, 1.0f, a.data(), K_M, b.data(), K_N, 0.0f,
                                            column_major_c.data(), K_M);

            for (int64_t row = 0; row < K_M; ++row)
                for (int64_t col = 0; col < K_N; ++col)
                    EXPECT_EQ(column_major_c[static_cast<std::size_t>(row + col * K_M)],
                              row_major_c[static_cast<std::size_t>(row * K_N + col)])
                        << "element (" << row << ", " << col << ")";
        }
    } // namespace Test
} // namespace SushiBLAS
