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
 * @file test_running_solution.cpp
 * @brief Tests RunningSolution: where it keeps a solved element and what it rounds.
 * @author Mustafa Garip
 */

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

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

        /// @brief The binary16 value nearest to K_INEXACT_IN_HALF: bits 0x2E66, 1.599609375 / 16.
        constexpr float K_NEAREST_HALF_TO_INEXACT = 0.0999755859375f;

        /// @brief Marks a workspace element no case may write.
        constexpr float K_UNTOUCHED = -7.0f;

        /// @brief The element types that are carried in themselves and need no workspace.
        using SelfCarriedTypes = ::testing::Types<float, std::complex<float>, std::complex<double>>;

        /// @brief Typed fixture for a RunningSolution over an element type carried in itself.
        template <typename T>
        class Small_RunningSolutionSelfCarried : public ::testing::Test
        {
        };
        SB_TYPED_TEST_SUITE(Small_RunningSolutionSelfCarried, SelfCarriedTypes);

        TYPED_TEST(Small_RunningSolutionSelfCarried, SetThenGetGoesThroughTheBoundArray)
        {
            using T = TypeParam;
            std::array<T, K_SOLUTION_LENGTH> bound = {T(1), T(2), T(3), T(4)};
            const Kernels::RunningSolution<T> solution(bound.data(), nullptr, 0);

            solution.set(1, T(9));

            EXPECT_EQ(bound, (std::array<T, K_SOLUTION_LENGTH>{T(1), T(9), T(3), T(4)}));
            EXPECT_EQ(solution.get(1), T(9));
        }

        TYPED_TEST(Small_RunningSolutionSelfCarried, GetSeesAWriteToTheBoundArray)
        {
            using T = TypeParam;
            std::array<T, K_SOLUTION_LENGTH> bound = {T(1), T(2), T(3), T(4)};
            const Kernels::RunningSolution<T> solution(bound.data(), nullptr, 0);

            bound[2] = T(8);

            EXPECT_EQ(solution.get(2), T(8));
        }

        TEST(Small_RunningSolution, FloatIgnoresTheWorkspaceItIsGiven)
        {
            std::array<float, K_SOLUTION_LENGTH> rounded = {1.0f, 2.0f, 3.0f, 4.0f};
            std::array<float, K_SOLUTION_LENGTH> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<float> solution(rounded.data(), workspace.data(), 0);

            solution.set(0, 5.0f);

            EXPECT_EQ(workspace, (std::array<float, K_SOLUTION_LENGTH>{K_UNTOUCHED, K_UNTOUCHED,
                                                                       K_UNTOUCHED, K_UNTOUCHED}));
        }

        TEST(Small_RunningSolution, HalfGetReturnsTheUnroundedFloat)
        {
            std::array<sycl::half, K_SOLUTION_LENGTH> rounded = {};
            std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<sycl::half> solution(rounded.data(), workspace.data(),
                                                                K_WORKSPACE_OFFSET);

            solution.set(1, K_INEXACT_IN_HALF);

            EXPECT_EQ(solution.get(1), K_INEXACT_IN_HALF);
        }

        TEST(Small_RunningSolution, HalfSetStoresTheRoundingInTheBoundArray)
        {
            std::array<sycl::half, K_SOLUTION_LENGTH> rounded = {};
            std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<sycl::half> solution(rounded.data(), workspace.data(),
                                                                K_WORKSPACE_OFFSET);

            solution.set(1, K_INEXACT_IN_HALF);

            EXPECT_EQ(static_cast<float>(rounded[1]), K_NEAREST_HALF_TO_INEXACT);
            EXPECT_EQ(static_cast<float>(rounded[0]), 0.0f);
        }

        TEST(Small_RunningSolution, HalfKeepsTheWideCopyAtTheWorkspaceOffset)
        {
            std::array<sycl::half, K_SOLUTION_LENGTH> rounded = {};
            std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> workspace;
            workspace.fill(K_UNTOUCHED);
            const Kernels::RunningSolution<sycl::half> solution(rounded.data(), workspace.data(),
                                                                K_WORKSPACE_OFFSET);

            solution.set(1, K_INEXACT_IN_HALF);

            // Element 1 of the solution sits at K_WORKSPACE_OFFSET + 1 = 3 of the workspace.
            const std::array<float, K_SOLUTION_LENGTH + K_WORKSPACE_OFFSET> expected = {
                K_UNTOUCHED, K_UNTOUCHED, K_UNTOUCHED, K_INEXACT_IN_HALF, K_UNTOUCHED, K_UNTOUCHED};
            EXPECT_EQ(workspace, expected);
        }
    } // namespace Test
} // namespace SushiBLAS
