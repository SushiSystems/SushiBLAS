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
 * @file test_syr2k.cpp
 * @brief Tests the reference syr2k, single and batched, against a dense rank-2k update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syr2k.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
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

        TEST(Unit_ReferenceSyr2k, MatchesADenseUpdateOnEachTriangle)
        {
            // A and B are n x k, row-major and not transposed.
            const std::vector<double> a = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const std::vector<double> b = small_integers<double>(K_ORDER * K_DEPTH, 2);
            const std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 3);
            for (const bool upper : {false, true})
            {
                SCOPED_TRACE(upper);
                std::vector<double> expected = c;
                for (int64_t i = 0; i < K_ORDER; ++i)
                {
                    for (int64_t j = 0; j < K_ORDER; ++j)
                    {
                        if (!(upper ? (i <= j) : (i >= j)))
                            continue;
                        double sum = 0.0;
                        for (int64_t p = 0; p < K_DEPTH; ++p)
                            sum += element(a, K_DEPTH, i, p) * element(b, K_DEPTH, j, p) +
                                   element(b, K_DEPTH, i, p) * element(a, K_DEPTH, j, p);
                        double& entry = expected[static_cast<std::size_t>(i * K_ORDER + j)];
                        entry = 3.0 * sum + 2.0 * entry;
                    }
                }
                std::vector<double> got = c;

                Kernels::Reference::syr2k(Core::Layout::RowMajor, uplo_of(upper),
                                          Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 3.0,
                                          a.data(), K_DEPTH, b.data(), K_DEPTH, 2.0, got.data(),
                                          K_ORDER);

                EXPECT_EQ(got, expected);
            }
        }

        TEST(Unit_ReferenceSyr2k, TheBatchedFormUpdatesEachMatrixOfTheBatch)
        {
            const int64_t batch = 3;
            const int64_t panel = K_ORDER * K_DEPTH;
            const std::vector<double> a = small_integers<double>(batch * panel, 4);
            const std::vector<double> b = small_integers<double>(batch * panel, 5);
            std::vector<double> batched = small_integers<double>(batch * K_ORDER * K_ORDER, 6);
            std::vector<double> one_by_one = batched;

            Kernels::Reference::syr2k(Core::Layout::ColumnMajor, Core::Uplo::Upper,
                                      Core::Transpose::Trans, K_ORDER, K_DEPTH, 1.0, a.data(),
                                      K_DEPTH, panel, b.data(), K_DEPTH, panel, 1.0,
                                      batched.data(), K_ORDER, K_ORDER * K_ORDER, batch);
            for (int64_t i = 0; i < batch; ++i)
                Kernels::Reference::syr2k(Core::Layout::ColumnMajor, Core::Uplo::Upper,
                                          Core::Transpose::Trans, K_ORDER, K_DEPTH, 1.0,
                                          a.data() + i * panel, K_DEPTH, b.data() + i * panel,
                                          K_DEPTH, 1.0,
                                          one_by_one.data() + i * K_ORDER * K_ORDER, K_ORDER);

            EXPECT_EQ(batched, one_by_one);
        }
    } // namespace Test
} // namespace SushiBLAS
