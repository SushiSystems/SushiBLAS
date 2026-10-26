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
 * @brief Tests that the SYCL GMRES equals the reference bit for bit on the host device.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/sycl/gmres.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/linalg_kernels/gmres_workspace.hpp>
#include <SushiBLAS/linalg_kernels/reference/gmres.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The order of every test system.
            constexpr std::size_t K_N = 300;

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

            /** @brief Holds what one device solve wrote. */
            struct DeviceSolve
            {
                    std::vector<double> x;        /**< The solution. */
                    std::uint32_t iterations = 0; /**< Inner iterations taken. */
                    double residual = 0.0;        /**< The reported relative residual. */
            };

            /** @brief Runs the SYCL solve on @p q from a zero guess. */
            DeviceSolve run_device(sycl::queue& q, const std::vector<double>& a,
                                   const std::vector<double>& b,
                                   const Kernels::GmresSettings& settings)
            {
                const SharedArray<double> matrix(q, a);
                const SharedArray<double> rhs(q, b);
                SharedArray<double> x(q, K_N);
                SharedArray<std::uint32_t> iterations(q, 1);
                SharedArray<double> residual(q, 1);
                const Kernels::GmresLayout layout(K_N, settings.restart, settings.block_size);
                SharedArray<unsigned char> workspace(q, layout.bytes(sizeof(double)));

                Kernels::SYCL::gmres(q, matrix.get(), rhs.get(), x.get(), iterations.get(),
                                     residual.get(), K_N, settings, workspace.get(), {})
                    .wait();

                DeviceSolve out;
                out.x = x.values();
                out.iterations = iterations.values()[0];
                out.residual = residual.values()[0];
                return out;
            }
        } // namespace

        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLGmres : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLGmres, EqualsTheReferenceWithRestartsAndBlocks)
        {
            std::vector<double> a = values(K_N * K_N, 41);
            for (std::size_t i = 0; i < K_N; ++i)
                a[i * K_N + i] += static_cast<double>(K_N);
            const std::vector<double> b = values(K_N, 42);
            Kernels::GmresSettings settings;
            settings.restart = 4;
            settings.block_size = 3;
            settings.relative_tolerance = 1e-11;

            std::vector<double> expected(K_N, 0.0);
            std::uint32_t expected_iterations = 0;
            double expected_residual = 0.0;
            Kernels::Reference::gmres(a.data(), b.data(), expected.data(), &expected_iterations,
                                      &expected_residual, K_N, settings);

            const DeviceSolve got = run_device(queue(), a, b, settings);

            EXPECT_LE(got.residual, 1e-11);
            EXPECT_EQ(got.iterations, expected_iterations);
            EXPECT_EQ(got.residual, expected_residual);
            EXPECT_EQ(got.x, expected);
        }

        TEST_F(Medium_SYCLGmres, TwoRunsAgreeBitForBit)
        {
            std::vector<double> a = values(K_N * K_N, 43);
            for (std::size_t i = 0; i < K_N; ++i)
                a[i * K_N + i] += static_cast<double>(K_N);
            const std::vector<double> b = values(K_N, 44);
            const Kernels::GmresSettings settings;

            const DeviceSolve first = run_device(queue(), a, b, settings);
            const DeviceSolve second = run_device(queue(), a, b, settings);

            EXPECT_EQ(first.x, second.x);
            EXPECT_EQ(first.iterations, second.iterations);
            EXPECT_EQ(first.residual, second.residual);
        }

        TEST_F(Medium_SYCLGmres, AZeroRightHandSideGivesZero)
        {
            std::vector<double> a = values(K_N * K_N, 45);
            for (std::size_t i = 0; i < K_N; ++i)
                a[i * K_N + i] += static_cast<double>(K_N);
            const Kernels::GmresSettings settings;

            const DeviceSolve got =
                run_device(queue(), a, std::vector<double>(K_N, 0.0), settings);

            EXPECT_EQ(got.x, std::vector<double>(K_N, 0.0));
            EXPECT_EQ(got.iterations, 0u);
            EXPECT_EQ(got.residual, 0.0);
        }
    } // namespace Test
} // namespace SushiBLAS
