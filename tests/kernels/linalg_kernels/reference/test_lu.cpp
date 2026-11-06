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
 * @file test_lu.cpp
 * @brief Tests the reference batched LU by P L U reconstruction, its pivot rule, its singular
 * flag and its solve residual.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/reference/lu.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Rounded operations per unit of the order n that the factor and the solve may spend.
            constexpr std::size_t K_OPERATIONS_PER_ORDER = 8;

            /// @brief One shape of a batched system.
            struct Shape
            {
                    std::size_t n;     /**< The order of each matrix. */
                    std::size_t rhs;   /**< The right-hand sides per matrix. */
                    std::size_t batch; /**< The matrices in the batch. */
            };

            /// @brief The orders the reconstruction is checked at: one, small and larger than a block.
            constexpr std::size_t K_RECONSTRUCTION_ORDERS[] = {1, 5, 40};

            /// @brief The shapes the solve residual is checked at.
            constexpr Shape K_SOLVE_SHAPES[] = {{2, 2, 1}, {12, 3, 2}};

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

            /** @brief Returns a batch of matrices whose diagonal outweighs each row, so the condition number stays near 1. */
            std::vector<double> dominant_batch(std::size_t batch, std::size_t n, unsigned seed)
            {
                std::vector<double> a = values(batch * n * n, seed);
                for (std::size_t b = 0; b < batch; ++b)
                    for (std::size_t i = 0; i < n; ++i)
                        a[b * n * n + i * n + i] += static_cast<double>(n);
                return a;
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

            /** @brief Expects P^T L U to give back every matrix of a batch of three, at each order of the table. */
            template<typename T>
            void expect_reconstructions()
            {
                constexpr std::size_t K_BATCH = 3;
                for (const std::size_t n : K_RECONSTRUCTION_ORDERS)
                {
                    SCOPED_TRACE(n);
                    const std::vector<double> original = values(K_BATCH * n * n, 11);
                    std::vector<T> a(original.begin(), original.end());
                    std::vector<std::uint32_t> pivots(K_BATCH * n, 99u);
                    std::vector<std::uint32_t> singular(K_BATCH, 7u);

                    Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), n,
                                                  K_BATCH);

                    const double bound = absolute_tolerance<T>(K_OPERATIONS_PER_ORDER * n);
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

            /** @brief Expects A X to give back B for a batch at each shape of the table. */
            template<typename T>
            void expect_small_residuals()
            {
                for (const Shape& shape : K_SOLVE_SHAPES)
                {
                    SCOPED_TRACE(shape.n);
                    const std::vector<double> original = dominant_batch(shape.batch, shape.n, 12);
                    const std::vector<double> b_values =
                        values(shape.batch * shape.n * shape.rhs, 13);
                    std::vector<T> a(original.begin(), original.end());
                    std::vector<T> x(b_values.begin(), b_values.end());
                    std::vector<std::uint32_t> pivots(shape.batch * shape.n);
                    std::vector<std::uint32_t> singular(shape.batch);

                    Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(),
                                                  shape.n, shape.batch);
                    Kernels::Reference::lu_solve(a.data(), pivots.data(), x.data(), shape.n,
                                                 shape.rhs, shape.batch);

                    const double bound = absolute_tolerance<T>(K_OPERATIONS_PER_ORDER * shape.n);
                    for (std::size_t b = 0; b < shape.batch; ++b)
                        for (std::size_t i = 0; i < shape.n; ++i)
                            for (std::size_t j = 0; j < shape.rhs; ++j)
                            {
                                double sum = 0.0;
                                for (std::size_t k = 0; k < shape.n; ++k)
                                    sum += static_cast<double>(static_cast<T>(
                                               original[b * shape.n * shape.n + i * shape.n + k])) *
                                           static_cast<double>(
                                               x[b * shape.n * shape.rhs + k * shape.rhs + j]);
                                EXPECT_NEAR(
                                    sum,
                                    static_cast<T>(
                                        b_values[b * shape.n * shape.rhs + i * shape.rhs + j]),
                                    bound)
                                    << b << " " << i << " " << j;
                            }
                }
            }

            /** @brief Holds what one factor of a batch wrote. */
            struct Factored
            {
                    std::vector<std::uint32_t> pivots;   /**< The pivot rows. */
                    std::vector<std::uint32_t> singular; /**< The singular flag of each matrix. */
            };

            /** @brief Factors a batch of matrices of order n, starting from sentinel pivots and flags. */
            template<typename T>
            Factored factor(std::vector<T> a, std::size_t n, std::size_t batch)
            {
                Factored out{std::vector<std::uint32_t>(batch * n, 99u),
                             std::vector<std::uint32_t>(batch, 7u)};
                Kernels::Reference::lu_factor(a.data(), out.pivots.data(), out.singular.data(), n,
                                              batch);
                return out;
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
            expect_reconstructions<TypeParam>();
        }

        TYPED_TEST(Small_ReferenceLU, PivotsOnTheLargestMagnitude)
        {
            using T = TypeParam;
            // Column 0 holds 1, 2 and -3; row 2 has the largest magnitude.
            const Factored f =
                factor<T>({T(1), T(2), T(0), T(2), T(1), T(1), T(-3), T(0), T(1)}, 3, 1);

            EXPECT_EQ(f.pivots[0], 2u);
        }

        TYPED_TEST(Small_ReferenceLU, PivotsOnTheLowestRowOnATie)
        {
            using T = TypeParam;
            // Column 0 holds |-3| at rows 1 and 2; row 1 wins the tie.
            const Factored f =
                factor<T>({T(1), T(2), T(0), T(-3), T(1), T(1), T(3), T(0), T(1)}, 3, 1);

            EXPECT_EQ(f.pivots[0], 1u);
        }

        TYPED_TEST(Small_ReferenceLU, FlagsASingularMatrix)
        {
            using T = TypeParam;
            // Matrix 0 has two equal rows; matrix 1 is the identity.
            const Factored f = factor<T>({T(1), T(2), T(1), T(2), T(1), T(0), T(0), T(1)}, 2, 2);

            EXPECT_EQ(f.singular[0], 1u);
        }

        TYPED_TEST(Small_ReferenceLU, LeavesTheOtherMatricesOfTheBatchUnflagged)
        {
            using T = TypeParam;
            // Matrix 0 has two equal rows; matrix 1 is the identity.
            const Factored f = factor<T>({T(1), T(2), T(1), T(2), T(1), T(0), T(0), T(1)}, 2, 2);

            EXPECT_EQ(f.singular[1], 0u);
            EXPECT_EQ(f.pivots[2], 0u);
            EXPECT_EQ(f.pivots[3], 1u);
        }

        TYPED_TEST(Small_ReferenceLU, FlagsTheZeroMatrix)
        {
            using T = TypeParam;
            const Factored f = factor<T>(std::vector<T>(9, T(0)), 3, 1);

            EXPECT_EQ(f.singular[0], 1u);
        }

        TYPED_TEST(Small_ReferenceLU, PivotsOnTheLowestRowWhenEveryCandidateIsZero)
        {
            using T = TypeParam;
            const Factored f = factor<T>(std::vector<T>(9, T(0)), 3, 1);

            EXPECT_EQ(f.pivots, (std::vector<std::uint32_t>{0u, 1u, 2u}));
        }

        TYPED_TEST(Small_ReferenceLU, SolveLeavesASmallResidualOnABatch)
        {
            expect_small_residuals<TypeParam>();
        }
    } // namespace Test
} // namespace SushiBLAS
