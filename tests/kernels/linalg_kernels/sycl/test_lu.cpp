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
 * @brief Tests that the SYCL batched LU and its solve equal the reference bit for bit.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/sycl/lu.hpp>

#include <string>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/linalg_kernels/reference/lu.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

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

            /** @brief Returns a batch whose matrix 2 repeats its first row, so a pivot vanishes once n > 1. */
            std::vector<double> batch_with_a_repeated_row(std::size_t batch, std::size_t n,
                                                          unsigned seed)
            {
                std::vector<double> a = values(batch * n * n, seed);
                if (n > 1)
                    std::copy(a.begin() + 2 * n * n, a.begin() + 2 * n * n + n,
                              a.begin() + 2 * n * n + n);
                return a;
            }

            /** @brief Returns a batch of matrices whose diagonal outweighs each row. */
            std::vector<double> dominant_batch(std::size_t batch, std::size_t n, unsigned seed)
            {
                std::vector<double> a = values(batch * n * n, seed);
                for (std::size_t b = 0; b < batch; ++b)
                    for (std::size_t i = 0; i < n; ++i)
                        a[b * n * n + i * n + i] += static_cast<double>(n);
                return a;
            }

            /** @brief One order the factor is run at. */
            struct FactorRow
            {
                    const char* name; /**< The row's suffix. */
                    std::size_t n;    /**< The order of each matrix. */
            };

            /** @brief Returns the orders: one, small and larger than a block. */
            std::vector<FactorRow> factor_rows()
            {
                return {{"OrderOne", 1}, {"OrderSeven", 7}, {"OrderSeventy", 70}};
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string factor_row_name(const ::testing::TestParamInfo<FactorRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the kernels on the host CPU queue. */
        class Medium_SYCLLU : public HostDeviceTest
        {

        };

        /** @brief Runs the factor on the host CPU queue, one order at a time. */
        class Medium_SYCLLUFactor : public HostDeviceTest, public ::testing::WithParamInterface<FactorRow>
        {

        };

        TEST_P(Medium_SYCLLUFactor, FactorEqualsTheReferenceWithASingularMatrixInTheBatch)
        {
            const std::size_t n = GetParam().n;
            constexpr std::size_t K_BATCH = 4;
            std::vector<double> a = batch_with_a_repeated_row(K_BATCH, n, 31);
            std::vector<double> expected = a;
            std::vector<std::uint32_t> expected_pivots(K_BATCH * n);
            std::vector<std::uint32_t> expected_singular(K_BATCH);
            Kernels::Reference::lu_factor(expected.data(), expected_pivots.data(),
                                          expected_singular.data(), n, K_BATCH);

            SharedArray<double> device_a(queue(), a);
            SharedArray<std::uint32_t> pivots(queue(), K_BATCH * n);
            SharedArray<std::uint32_t> singular(queue(), K_BATCH);
            Kernels::SYCL::lu_factor(queue(), device_a.get(), pivots.get(), singular.get(), n,
                                     K_BATCH, {})
                .wait();

            EXPECT_EQ(singular.values(), expected_singular);
            EXPECT_EQ(pivots.values(), expected_pivots);
            EXPECT_EQ(device_a.values(), expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLLUFactor, ::testing::ValuesIn(factor_rows()),
                                 factor_row_name);

        TEST_F(Medium_SYCLLU, SolveEqualsTheReference)
        {
            constexpr std::size_t K_N = 9;
            constexpr std::size_t K_RHS = 3;
            constexpr std::size_t K_BATCH = 2;
            std::vector<double> a = dominant_batch(K_BATCH, K_N, 32);
            std::vector<std::uint32_t> pivots(K_BATCH * K_N);
            std::vector<std::uint32_t> singular(K_BATCH);
            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), K_N, K_BATCH);
            const std::vector<double> rhs = values(K_BATCH * K_N * K_RHS, 33);
            std::vector<double> expected = rhs;
            Kernels::Reference::lu_solve(a.data(), pivots.data(), expected.data(), K_N, K_RHS,
                                         K_BATCH);

            const SharedArray<double> factors(queue(), a);
            const SharedArray<std::uint32_t> device_pivots(queue(), pivots);
            SharedArray<double> x(queue(), rhs);
            Kernels::SYCL::lu_solve(queue(), factors.get(), device_pivots.get(), x.get(), K_N,
                                    K_RHS, K_BATCH, {})
                .wait();

            EXPECT_EQ(x.values(), expected);
        }
    } // namespace Test
} // namespace SushiBLAS
