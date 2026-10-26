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
 * @brief Tests that the SYCL batched LU and its solve equal the reference bit for bit.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/sycl/lu.hpp>

#include <vector>
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
        } // namespace

        /** @brief Runs the kernels on the host CPU queue. */
        class Medium_SYCLLU : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLLU, FactorEqualsTheReferenceWithASingularMatrixInTheBatch)
        {
            for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{70}})
            {
                SCOPED_TRACE(n);
                constexpr std::size_t K_BATCH = 4;
                std::vector<double> a = values(K_BATCH * n * n, 31);
                // Matrix 2 repeats its first row, so a pivot vanishes once n > 1.
                if (n > 1)
                    for (std::size_t j = 0; j < n; ++j)
                        a[2 * n * n + n + j] = a[2 * n * n + j];
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
        }

        TEST_F(Medium_SYCLLU, SolveEqualsTheReference)
        {
            constexpr std::size_t K_N = 9;
            constexpr std::size_t K_RHS = 3;
            constexpr std::size_t K_BATCH = 2;
            std::vector<double> a = values(K_BATCH * K_N * K_N, 32);
            for (std::size_t b = 0; b < K_BATCH; ++b)
                for (std::size_t i = 0; i < K_N; ++i)
                    a[b * K_N * K_N + i * K_N + i] += static_cast<double>(K_N);
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
