/**************************************************************************/
/* test_lu.cpp                                                            */
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
 * @file test_lu.cpp
 * @brief Tests the reference batched LU by P L U reconstruction, its pivot rule, its singular
 * flag and its solve.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/reference/lu.hpp>

#include <limits>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns count fixed pseudo-random values in [-1, 1) from a seed. */
            std::vector<double> values(std::size_t count, unsigned seed)
            {
                std::vector<double> out(count);
                unsigned state = seed * 2654435761u + 1u;
                for (double& v : out)
                {
                    state = state * 1664525u + 1013904223u;
                    v = static_cast<double>(state >> 8) / static_cast<double>(1u << 24) * 2.0 - 1.0;
                }
                return out;
            }

            /**
             * @brief Returns P^T L U of one factored matrix, which equals the original A.
             * @param lu Unit-diagonal L below the diagonal and U on and above it.
             */
            template<typename T>
            std::vector<double> reconstruct(const T* lu, const std::uint32_t* pivots,
                                            std::size_t n)
            {
                std::vector<double> product(n * n, 0.0);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        double sum = 0.0;
                        for (std::size_t k = 0; k <= std::min(i, j); ++k)
                        {
                            const double l = k == i ? 1.0 : static_cast<double>(lu[i * n + k]);
                            sum += l * static_cast<double>(lu[k * n + j]);
                        }
                        product[i * n + j] = sum;
                    }
                // The swaps were applied to A in step order, so they are undone in reverse.
                for (std::size_t k = n; k-- > 0;)
                    for (std::size_t j = 0; j < n; ++j)
                        std::swap(product[k * n + j], product[pivots[k] * n + j]);
                return product;
            }
        } // namespace

        /** @brief Runs the reference LU in one scalar type. */
        template<typename T>
        class Small_ReferenceLU : public ::testing::Test
        {

        };

        using LUScalars = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_ReferenceLU, LUScalars);

        TYPED_TEST(Small_ReferenceLU, PLUReconstructsEveryMatrixOfTheBatch)
        {
            using T = TypeParam;
            for (const std::size_t n : {std::size_t{1}, std::size_t{5}, std::size_t{40}})
            {
                SCOPED_TRACE(n);
                constexpr std::size_t K_BATCH = 3;
                const std::vector<double> original = values(K_BATCH * n * n, 11);
                std::vector<T> a(original.begin(), original.end());
                std::vector<std::uint32_t> pivots(K_BATCH * n, 99u);
                std::vector<std::uint32_t> singular(K_BATCH, 7u);

                Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), n,
                                              K_BATCH);

                const double bound = 64.0 * static_cast<double>(n) *
                                     std::numeric_limits<T>::epsilon();
                for (std::size_t b = 0; b < K_BATCH; ++b)
                {
                    EXPECT_EQ(singular[b], 0u) << b;
                    const std::vector<double> product =
                        reconstruct(a.data() + b * n * n, pivots.data() + b * n, n);
                    for (std::size_t e = 0; e < n * n; ++e)
                        EXPECT_NEAR(product[e], static_cast<T>(original[b * n * n + e]), bound)
                            << b << " " << e;
                }
            }
        }

        TYPED_TEST(Small_ReferenceLU, PivotsOnTheLargestMagnitudeAndTheLowestRowOnATie)
        {
            using T = TypeParam;
            // Column 0 holds |-3| at rows 1 and 2; row 1 wins the tie.
            std::vector<T> a{T(1), T(2), T(0), T(-3), T(1), T(1), T(3), T(0), T(1)};
            std::vector<std::uint32_t> pivots(3);
            std::vector<std::uint32_t> singular(1);

            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), 3, 1);

            EXPECT_EQ(pivots[0], 1u);
            EXPECT_EQ(singular[0], 0u);
        }

        TYPED_TEST(Small_ReferenceLU, FlagsASingularMatrixAndLeavesTheOthers)
        {
            using T = TypeParam;
            // Matrix 0 has two equal rows; matrix 1 is the identity.
            std::vector<T> a{T(1), T(2), T(1), T(2), T(1), T(0), T(0), T(1)};
            std::vector<std::uint32_t> pivots(4, 99u);
            std::vector<std::uint32_t> singular(2, 7u);

            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), 2, 2);

            EXPECT_EQ(singular[0], 1u);
            EXPECT_EQ(pivots[1], 1u);
            EXPECT_EQ(singular[1], 0u);
            EXPECT_EQ(pivots[2], 0u);
            EXPECT_EQ(pivots[3], 1u);
        }

        TYPED_TEST(Small_ReferenceLU, FlagsTheZeroMatrix)
        {
            using T = TypeParam;
            std::vector<T> a(9, T(0));
            std::vector<std::uint32_t> pivots(3, 99u);
            std::vector<std::uint32_t> singular(1, 7u);

            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), 3, 1);

            EXPECT_EQ(singular[0], 1u);
            EXPECT_EQ(pivots, (std::vector<std::uint32_t>{0u, 1u, 2u}));
        }

        TYPED_TEST(Small_ReferenceLU, SolveGivesTheHandWorkedSolution)
        {
            using T = TypeParam;
            // [2 1; 1 3] X = [3 5; 4 10] has X = [1 1; 1 3].
            std::vector<T> a{T(2), T(1), T(1), T(3)};
            std::vector<T> rhs{T(3), T(5), T(4), T(10)};
            std::vector<std::uint32_t> pivots(2);
            std::vector<std::uint32_t> singular(1);

            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), 2, 1);
            Kernels::Reference::lu_solve(a.data(), pivots.data(), rhs.data(), 2, 2, 1);

            const double bound = 8.0 * std::numeric_limits<T>::epsilon();
            EXPECT_NEAR(rhs[0], 1.0, bound);
            EXPECT_NEAR(rhs[1], 1.0, bound);
            EXPECT_NEAR(rhs[2], 1.0, bound);
            EXPECT_NEAR(rhs[3], 3.0, bound * 3.0);
        }

        TYPED_TEST(Small_ReferenceLU, SolveLeavesASmallResidualOnABatch)
        {
            using T = TypeParam;
            constexpr std::size_t K_N = 12;
            constexpr std::size_t K_RHS = 3;
            constexpr std::size_t K_BATCH = 2;
            std::vector<double> original = values(K_BATCH * K_N * K_N, 12);
            // A dominant diagonal keeps the condition number near 1.
            for (std::size_t b = 0; b < K_BATCH; ++b)
                for (std::size_t i = 0; i < K_N; ++i)
                    original[b * K_N * K_N + i * K_N + i] += static_cast<double>(K_N);
            const std::vector<double> b_values = values(K_BATCH * K_N * K_RHS, 13);
            std::vector<T> a(original.begin(), original.end());
            std::vector<T> x(b_values.begin(), b_values.end());
            std::vector<std::uint32_t> pivots(K_BATCH * K_N);
            std::vector<std::uint32_t> singular(K_BATCH);

            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), K_N, K_BATCH);
            Kernels::Reference::lu_solve(a.data(), pivots.data(), x.data(), K_N, K_RHS, K_BATCH);

            const double bound = 64.0 * K_N * std::numeric_limits<T>::epsilon();
            for (std::size_t b = 0; b < K_BATCH; ++b)
                for (std::size_t i = 0; i < K_N; ++i)
                    for (std::size_t j = 0; j < K_RHS; ++j)
                    {
                        double sum = 0.0;
                        for (std::size_t k = 0; k < K_N; ++k)
                            sum += static_cast<double>(static_cast<T>(
                                       original[b * K_N * K_N + i * K_N + k])) *
                                   static_cast<double>(x[b * K_N * K_RHS + k * K_RHS + j]);
                        EXPECT_NEAR(sum,
                                    static_cast<T>(b_values[b * K_N * K_RHS + i * K_RHS + j]),
                                    bound)
                            << b << " " << i << " " << j;
                    }
        }
    } // namespace Test
} // namespace SushiBLAS
