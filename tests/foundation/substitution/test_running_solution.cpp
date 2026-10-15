/**************************************************************************/
/* test_running_solution.cpp                                              */
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
 * @file test_running_solution.cpp
 * @brief Tests RunningSolution: where it keeps a solved element and what it rounds.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Elements of the right-hand side every case binds.
        constexpr std::size_t K_SOLUTION_LENGTH = 4;

        /// @brief Elements of the workspace that precede the wide copy of the right-hand side.
        constexpr int64_t K_WORKSPACE_OFFSET = 2;

        /// @brief A binary32 value HALF cannot hold exactly.
        constexpr float K_INEXACT_IN_HALF = 0.1f;

        /// @brief Marks a workspace element no case may write.
        constexpr float K_UNTOUCHED = -7.0f;

        TEST(Medium_RunningSolution, OnlyHalfNeedsAWorkspace)
        {
            static_assert(Kernels::RunningSolution<sycl::half>::K_NEEDS_WORKSPACE,
                          "HALF is carried in FLOAT32.");
            static_assert(!Kernels::RunningSolution<float>::K_NEEDS_WORKSPACE,
                          "FLOAT32 is carried in itself.");
            static_assert(!Kernels::RunningSolution<double>::K_NEEDS_WORKSPACE,
                          "FLOAT64 is carried in itself.");
            static_assert(!Kernels::RunningSolution<std::complex<float>>::K_NEEDS_WORKSPACE,
                          "COMPLEX32 is carried in itself.");
            static_assert(!Kernels::RunningSolution<std::complex<double>>::K_NEEDS_WORKSPACE,
                          "COMPLEX64 is carried in itself.");

            EXPECT_TRUE(Kernels::RunningSolution<sycl::half>::K_NEEDS_WORKSPACE);
            EXPECT_FALSE(Kernels::RunningSolution<float>::K_NEEDS_WORKSPACE);
            EXPECT_FALSE(Kernels::RunningSolution<double>::K_NEEDS_WORKSPACE);
            EXPECT_FALSE(Kernels::RunningSolution<std::complex<float>>::K_NEEDS_WORKSPACE);
            EXPECT_FALSE(Kernels::RunningSolution<std::complex<double>>::K_NEEDS_WORKSPACE);
        }

        TEST(Medium_RunningSolution, HalfIsCarriedInFloat)
        {
            static_assert(std::is_same_v<Kernels::RunningSolution<sycl::half>::Wide, float>,
                          "HALF is carried in FLOAT32.");
            static_assert(std::is_same_v<Kernels::RunningSolution<double>::Wide, double>,
                          "FLOAT64 is carried in itself.");
            SUCCEED();
        }

        TEST(Medium_RunningSolution, FloatSetWritesTheBoundArray)
        {
            std::array<float, K_SOLUTION_LENGTH> rounded = {1.0f, 2.0f, 3.0f, 4.0f};
            const Kernels::RunningSolution<float> solution(rounded.data(), nullptr, 0);

            solution.set(1, K_INEXACT_IN_HALF);

            EXPECT_EQ(rounded[1], K_INEXACT_IN_HALF);
            EXPECT_EQ(rounded[0], 1.0f);
            EXPECT_EQ(rounded[2], 3.0f);
        }

        TEST(Medium_RunningSolution, FloatGetReadsTheBoundArray)
        {
            std::array<float, K_SOLUTION_LENGTH> rounded = {1.0f, 2.0f, 3.0f, 4.0f};
            const Kernels::RunningSolution<float> solution(rounded.data(), nullptr, 0);

            EXPECT_EQ(solution.get(2), 3.0f);

            rounded[2] = 9.0f;

            EXPECT_EQ(solution.get(2), 9.0f);
        }

        TEST(Medium_RunningSolution, FloatIgnoresTheWorkspaceItIsGiven)
        {
            std::array<float, K_SOLUTION_LENGTH> rounded = {1.0f, 2.0f, 3.0f, 4.0f};
            std::array<float, K_SOLUTION_LENGTH> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<float> solution(rounded.data(), workspace.data(), 0);

            solution.set(0, 5.0f);

            EXPECT_EQ(solution.get(0), 5.0f);
            EXPECT_EQ(rounded[0], 5.0f);
            EXPECT_EQ(workspace[0], K_UNTOUCHED);
        }

        TEST(Medium_RunningSolution, ComplexSetThenGetGoesThroughTheBoundArray)
        {
            std::array<std::complex<double>, K_SOLUTION_LENGTH> rounded = {};
            const Kernels::RunningSolution<std::complex<double>> solution(rounded.data(), nullptr,
                                                                          0);

            solution.set(3, std::complex<double>(1.5, -2.5));

            EXPECT_EQ(rounded[3], std::complex<double>(1.5, -2.5));
            EXPECT_EQ(solution.get(3), std::complex<double>(1.5, -2.5));
        }

        TEST(Medium_RunningSolution, HalfGetReturnsTheUnroundedFloat)
        {
            std::array<sycl::half, K_SOLUTION_LENGTH> rounded = {};
            std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<sycl::half> solution(rounded.data(), workspace.data(),
                                                                K_WORKSPACE_OFFSET);

            solution.set(1, K_INEXACT_IN_HALF);

            EXPECT_EQ(solution.get(1), K_INEXACT_IN_HALF);
            EXPECT_NE(static_cast<float>(rounded[1]), K_INEXACT_IN_HALF);
        }

        TEST(Medium_RunningSolution, HalfSetStoresTheRoundingInTheBoundArray)
        {
            std::array<sycl::half, K_SOLUTION_LENGTH> rounded = {};
            std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<sycl::half> solution(rounded.data(), workspace.data(),
                                                                K_WORKSPACE_OFFSET);

            solution.set(1, K_INEXACT_IN_HALF);

            EXPECT_EQ(static_cast<float>(rounded[1]),
                      static_cast<float>(static_cast<sycl::half>(K_INEXACT_IN_HALF)));
            EXPECT_EQ(static_cast<float>(rounded[0]), 0.0f);
        }

        TEST(Medium_RunningSolution, HalfKeepsTheWideCopyAtTheWorkspaceOffset)
        {
            std::array<sycl::half, K_SOLUTION_LENGTH> rounded = {};
            std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<sycl::half> solution(rounded.data(), workspace.data(),
                                                                K_WORKSPACE_OFFSET);

            solution.set(1, K_INEXACT_IN_HALF);

            const std::size_t written = static_cast<std::size_t>(K_WORKSPACE_OFFSET) + 1;
            for (std::size_t i = 0; i < workspace.size(); ++i)
                EXPECT_EQ(workspace[i], i == written ? K_INEXACT_IN_HALF : K_UNTOUCHED)
                    << "workspace element " << i;
        }
    } // namespace Test
} // namespace SushiBLAS
