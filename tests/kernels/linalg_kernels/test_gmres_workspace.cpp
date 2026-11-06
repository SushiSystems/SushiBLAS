/**************************************************************************/
/* test_gmres_workspace.cpp                                               */
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
 * @file test_gmres_workspace.cpp
 * @brief Tests that GmresLayout places every array of the workspace after the one before it, and that
 * a solve with the default GmresSettings reaches the default tolerance.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/linalg_kernels/reference/gmres.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a row-major matrix of order n whose diagonal, n, outweighs the off-diagonal sum of each row. */
            std::vector<double> diagonally_dominant_matrix(std::size_t n)
            {
                std::vector<double> a(n * n);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        a[i * n + j] = i == j ? static_cast<double>(n)
                                              : static_cast<double>((i * 7 + j * 3) % 5) - 2.0;
                return a;
            }
        } // namespace

        TEST(Small_GmresLayout, PlacesTheArraysBackToBack)
        {
            const Kernels::GmresLayout layout(10, 4, 2);

            EXPECT_EQ(layout.basis, 0u);
            EXPECT_EQ(layout.hessenberg, 5u * 10u);
            EXPECT_EQ(layout.cosines, layout.hessenberg + 5u * 4u);
            EXPECT_EQ(layout.sines, layout.cosines + 4u);
            EXPECT_EQ(layout.projection, layout.sines + 4u);
            EXPECT_EQ(layout.coefficients, layout.projection + 5u);
            EXPECT_EQ(layout.residual, layout.coefficients + 4u);
            EXPECT_EQ(layout.work, layout.residual + 10u);
            EXPECT_EQ(layout.preconditioned, layout.work + 10u);
            EXPECT_EQ(layout.factors, layout.preconditioned + 10u);
            EXPECT_EQ(layout.partials, layout.factors + 10u * 2u);
            EXPECT_EQ(layout.scalars, layout.partials + Kernels::dot_chunks(10u));
        }

        TEST(Small_GmresLayout, CountsThePivotsAfterTheScalars)
        {
            const Kernels::GmresLayout layout(6, 2, 3);

            EXPECT_EQ(layout.bytes(sizeof(double)),
                      layout.scalars * sizeof(double) + 6u * sizeof(std::uint32_t));
        }

        TEST(Small_GmresSettings, ADefaultSolveConvergesToTheDefaultTolerance)
        {
            constexpr std::size_t K_N = 12;
            const std::vector<double> a = diagonally_dominant_matrix(K_N);
            const std::vector<double> b(K_N, 1.0);
            std::vector<double> x(K_N, 0.0);
            std::uint32_t iterations = 0;
            double residual = -1.0;
            const Kernels::GmresSettings settings;

            Kernels::Reference::gmres(a.data(), b.data(), x.data(), &iterations, &residual, K_N,
                                      settings);

            EXPECT_GE(iterations, 1u);
            EXPECT_LT(iterations, settings.max_iterations);
            EXPECT_LE(residual, settings.relative_tolerance);
        }
    } // namespace Test
} // namespace SushiBLAS
