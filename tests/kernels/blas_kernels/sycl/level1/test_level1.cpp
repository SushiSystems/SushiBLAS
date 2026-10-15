/**************************************************************************/
/* test_level1.cpp                                                        */
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
 * @file test_level1.cpp
 * @brief Tests the SYCL Level 1 kernels against the reference kernels on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/level1.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/level1.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace Ref = Kernels::Reference;

        /// @brief The vector length; large enough for the reductions to stage partials.
        constexpr int64_t K_LENGTH = 40000;

        /// @brief The scratch the staged reductions write their partials into.
        constexpr std::size_t K_SCRATCH_BYTES = 4096;

        /** @brief Runs the Level 1 kernels on the host CPU queue. */
        class Medium_SYCLLevel1 : public HostDeviceTest
        {
            protected:
                /** @brief Returns the operand of the given seed. */
                static std::vector<float> operand(std::uint32_t seed)
                {
                    return small_integers<float>(static_cast<std::size_t>(K_LENGTH), seed);
                }
        };

        TEST_F(Medium_SYCLLevel1, TheElementwiseKernelsMatchTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const std::vector<float> x = operand(1);
            std::vector<float> y = operand(2);
            SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), y);

            Kernels::SYCL::axpy<float>(queue(), K_LENGTH, 3.0f, device_x.get(), 1,
                                       device_y.get(), 1, {}).wait();
            Kernels::SYCL::scal<float>(queue(), K_LENGTH, -2.0f, device_y.get(), 1, {}).wait();
            Kernels::SYCL::rot<float>(queue(), K_LENGTH, device_x.get(), 1, device_y.get(), 1,
                                      0.0f, 1.0f, {}).wait();
            Ref::axpy<float>(K_LENGTH, 3.0f, x.data(), 1, y.data(), 1);
            Ref::scal<float>(K_LENGTH, -2.0f, y.data(), 1);
            std::vector<float> rotated_x = x;
            Ref::rot<float>(K_LENGTH, rotated_x.data(), 1, y.data(), 1, 0.0f, 1.0f);

            EXPECT_EQ(device_x.values(), rotated_x);
            EXPECT_EQ(device_y.values(), y);
        }

        TEST_F(Medium_SYCLLevel1, CopyAndSwapMoveEveryOtherElement)
        {
            const std::vector<float> x = operand(3);
            SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), static_cast<std::size_t>(K_LENGTH / 2));

            Kernels::SYCL::copy<float>(queue(), K_LENGTH / 2, device_x.get(), 2, device_y.get(),
                                       1, {}).wait();
            std::vector<float> expected(static_cast<std::size_t>(K_LENGTH / 2));
            Ref::copy<float>(K_LENGTH / 2, x.data(), 2, expected.data(), 1);
            EXPECT_EQ(device_y.values(), expected);

            Kernels::SYCL::swap<float>(queue(), K_LENGTH / 2, device_x.get(), 2, device_y.get(),
                                       1, {}).wait();
            EXPECT_EQ(device_y.values(), expected);
        }

        TEST_F(Medium_SYCLLevel1, TheReductionsMatchTheReferenceExactly)
        {
            const std::vector<double> x = small_integers<double>(K_LENGTH, 4);
            const std::vector<double> y = small_integers<double>(K_LENGTH, 5);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_y(queue(), y);
            SharedArray<double> results(queue(), 2);
            SharedArray<int64_t> index(queue(), 1);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);

            Kernels::SYCL::asum<double>(queue(), K_LENGTH, device_x.get(), 1, results.get(),
                                        scratch.get(), K_SCRATCH_BYTES, {}).wait();
            Kernels::SYCL::dot<double>(queue(), K_LENGTH, device_x.get(), 1, device_y.get(), 1,
                                       results.get() + 1, scratch.get(), K_SCRATCH_BYTES,
                                       {}).wait();
            Kernels::SYCL::iamax<double>(queue(), K_LENGTH, device_x.get(), 1, index.get(),
                                         scratch.get(), K_SCRATCH_BYTES, {}).wait();
            double asum = 0.0;
            double dot = 0.0;
            int64_t first_largest = -1;
            Ref::asum<double>(K_LENGTH, x.data(), 1, &asum);
            Ref::dot<double>(K_LENGTH, x.data(), 1, y.data(), 1, &dot);
            Ref::iamax<double>(K_LENGTH, x.data(), 1, &first_largest);

            EXPECT_EQ(results.get()[0], asum);
            EXPECT_EQ(results.get()[1], dot);
            EXPECT_EQ(index.get()[0], first_largest);
        }

        TEST_F(Medium_SYCLLevel1, Nrm2StaysWithinTheBoundOfAScaledSum)
        {
            const std::vector<double> x = small_integers<double>(K_LENGTH, 6);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> norm(queue(), 1);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);

            Kernels::SYCL::nrm2<double>(queue(), K_LENGTH, device_x.get(), 1, norm.get(),
                                        scratch.get(), K_SCRATCH_BYTES, {}).wait();
            double expected = 0.0;
            Ref::nrm2<double>(K_LENGTH, x.data(), 1, &expected);

            // A staged sum of K_LENGTH squares may differ from a serial one by K_LENGTH ulps.
            const double bound =
                expected * static_cast<double>(K_LENGTH) * std::numeric_limits<double>::epsilon();
            EXPECT_NEAR(norm.get()[0], expected, bound);
        }

        TEST_F(Medium_SYCLLevel1, AnEmptyVectorSumsToZero)
        {
            SharedArray<float> result(queue(), std::vector<float>{7.0f});
            const SharedArray<float> x(queue(), 1);

            Kernels::SYCL::asum<float>(queue(), 0, x.get(), 1, result.get(), nullptr, 0,
                                       {}).wait();

            EXPECT_EQ(result.get()[0], 0.0f);
        }
    } // namespace Test
} // namespace SushiBLAS
