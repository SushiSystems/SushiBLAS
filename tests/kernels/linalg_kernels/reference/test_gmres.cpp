/**************************************************************************/
/* test_gmres.cpp                                                         */
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
 * @file test_gmres.cpp
 * @brief Tests the reference GMRES against a direct solve, its reported residual, its warm start
 * and its zero right-hand side.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/reference/gmres.hpp>

#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The order of every test system.
            constexpr std::size_t K_N = 24;

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

            /** @brief Returns a non-symmetric matrix whose diagonal outweighs each row. */
            std::vector<double> dominant_matrix(unsigned seed)
            {
                std::vector<double> a = values(K_N * K_N, seed);
                for (std::size_t i = 0; i < K_N; ++i)
                    a[i * K_N + i] += static_cast<double>(K_N);
                return a;
            }

            /** @brief Returns the solution of A x = b by Gaussian elimination in long double. */
            std::vector<double> direct_solve(const std::vector<double>& a,
                                             const std::vector<double>& b)
            {
                std::vector<long double> m(a.begin(), a.end());
                std::vector<long double> x(b.begin(), b.end());
                for (std::size_t k = 0; k < K_N; ++k)
                {
                    std::size_t best = k;
                    for (std::size_t i = k + 1; i < K_N; ++i)
                        if (std::fabs(m[i * K_N + k]) > std::fabs(m[best * K_N + k]))
                            best = i;
                    for (std::size_t j = 0; j < K_N; ++j)
                        std::swap(m[k * K_N + j], m[best * K_N + j]);
                    std::swap(x[k], x[best]);
                    for (std::size_t i = k + 1; i < K_N; ++i)
                    {
                        const long double f = m[i * K_N + k] / m[k * K_N + k];
                        for (std::size_t j = k; j < K_N; ++j)
                            m[i * K_N + j] -= f * m[k * K_N + j];
                        x[i] -= f * x[k];
                    }
                }
                for (std::size_t i = K_N; i-- > 0;)
                {
                    for (std::size_t j = i + 1; j < K_N; ++j)
                        x[i] -= m[i * K_N + j] * x[j];
                    x[i] /= m[i * K_N + i];
                }
                return std::vector<double>(x.begin(), x.end());
            }

            /** @brief Returns |b - A x| / |b| in double. */
            double true_residual(const std::vector<double>& a, const std::vector<double>& b,
                                 const std::vector<double>& x)
            {
                double r = 0.0;
                double norm = 0.0;
                for (std::size_t i = 0; i < K_N; ++i)
                {
                    double ax = 0.0;
                    for (std::size_t j = 0; j < K_N; ++j)
                        ax += a[i * K_N + j] * x[j];
                    r += (b[i] - ax) * (b[i] - ax);
                    norm += b[i] * b[i];
                }
                return std::sqrt(r) / std::sqrt(norm);
            }

            /** @brief Holds what one solve wrote. */
            struct Solve
            {
                    std::vector<double> x;       /**< The solution. */
                    std::uint32_t iterations = 0; /**< Inner iterations taken. */
                    double residual = 0.0;       /**< The reported relative residual. */
            };

            /** @brief Runs the reference solve from @p guess. */
            Solve run(const std::vector<double>& a, const std::vector<double>& b,
                      std::vector<double> guess, const Kernels::GmresSettings& settings)
            {
                Solve out;
                double residual = -1.0;
                Kernels::Reference::gmres(a.data(), b.data(), guess.data(), &out.iterations,
                                          &residual, K_N, settings);
                out.x = std::move(guess);
                out.residual = residual;
                return out;
            }
        } // namespace

        TEST(Small_ReferenceGmres, ConvergesToTheDirectSolution)
        {
            const std::vector<double> a = dominant_matrix(21);
            const std::vector<double> b = values(K_N, 22);
            Kernels::GmresSettings settings;
            settings.relative_tolerance = 1e-12;

            const Solve s = run(a, b, std::vector<double>(K_N, 0.0), settings);

            const std::vector<double> expected = direct_solve(a, b);
            EXPECT_LE(s.residual, 1e-12);
            EXPECT_GE(s.iterations, 1u);
            for (std::size_t i = 0; i < K_N; ++i)
                EXPECT_NEAR(s.x[i], expected[i], 1e-10) << i;
        }

        TEST(Small_ReferenceGmres, RestartsAndBlocksStillConverge)
        {
            const std::vector<double> a = dominant_matrix(23);
            const std::vector<double> b = values(K_N, 24);
            Kernels::GmresSettings settings;
            settings.restart = 3;
            settings.block_size = 4;
            settings.relative_tolerance = 1e-10;

            const Solve s = run(a, b, std::vector<double>(K_N, 0.0), settings);

            EXPECT_LE(s.residual, 1e-10);
            // Three vectors per cycle cannot reach 1e-10 here in one cycle.
            EXPECT_GT(s.iterations, 3u);
            EXPECT_NEAR(s.residual, true_residual(a, b, s.x), 1e-14);
        }

        TEST(Small_ReferenceGmres, ReportsTheTrueResidualWhenTheCapStopsIt)
        {
            const std::vector<double> a = dominant_matrix(25);
            const std::vector<double> b = values(K_N, 26);
            Kernels::GmresSettings settings;
            settings.restart = 2;
            settings.max_iterations = 2;
            settings.relative_tolerance = 1e-14;

            const Solve s = run(a, b, std::vector<double>(K_N, 0.0), settings);

            EXPECT_EQ(s.iterations, 2u);
            EXPECT_GT(s.residual, 1e-14);
            EXPECT_NEAR(s.residual, true_residual(a, b, s.x), 1e-14);
        }

        TEST(Small_ReferenceGmres, AnExactWarmStartTakesNoIteration)
        {
            const std::vector<double> a = dominant_matrix(27);
            const std::vector<double> b = values(K_N, 28);
            Kernels::GmresSettings settings;
            settings.relative_tolerance = 1e-6;

            const Solve s = run(a, b, direct_solve(a, b), settings);

            EXPECT_EQ(s.iterations, 0u);
            EXPECT_LE(s.residual, 1e-6);
        }

        TEST(Small_ReferenceGmres, AZeroRightHandSideGivesZero)
        {
            const std::vector<double> a = dominant_matrix(29);
            const Kernels::GmresSettings settings;

            const Solve s = run(a, std::vector<double>(K_N, 0.0), std::vector<double>(K_N, 3.0),
                                settings);

            EXPECT_EQ(s.x, std::vector<double>(K_N, 0.0));
            EXPECT_EQ(s.iterations, 0u);
            EXPECT_EQ(s.residual, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
