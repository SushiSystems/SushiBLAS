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
 * @file test_level1.cpp
 * @brief Tests the SYCL Level 1 kernels against the reference kernels on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/level1.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/level1.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "support/tolerance.hpp"

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

        /// @brief Tests that the SYCL axpy matches the reference; the values are small integers, so no rounding.
        TEST_F(Medium_SYCLLevel1, AxpyMatchesTheReference)
        {
            const std::vector<float> x = operand(1);
            std::vector<float> y = operand(2);
            SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), y);

            Kernels::SYCL::axpy<float>(queue(), K_LENGTH, 3.0f, device_x.get(), 1,
                                       device_y.get(), 1, {}).wait();
            Ref::axpy<float>(K_LENGTH, 3.0f, x.data(), 1, y.data(), 1);

            EXPECT_EQ(device_y.values(), y);
        }

        /// @brief Tests that the SYCL scal matches the reference.
        TEST_F(Medium_SYCLLevel1, ScalMatchesTheReference)
        {
            std::vector<float> y = operand(2);
            SharedArray<float> device_y(queue(), y);

            Kernels::SYCL::scal<float>(queue(), K_LENGTH, -2.0f, device_y.get(), 1, {}).wait();
            Ref::scal<float>(K_LENGTH, -2.0f, y.data(), 1);

            EXPECT_EQ(device_y.values(), y);
        }

        /// @brief Tests that the SYCL rot matches the reference on both vectors.
        TEST_F(Medium_SYCLLevel1, RotMatchesTheReference)
        {
            std::vector<float> x = operand(1);
            std::vector<float> y = operand(2);
            SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), y);

            Kernels::SYCL::rot<float>(queue(), K_LENGTH, device_x.get(), 1, device_y.get(), 1,
                                      0.0f, 1.0f, {}).wait();
            Ref::rot<float>(K_LENGTH, x.data(), 1, y.data(), 1, 0.0f, 1.0f);

            EXPECT_EQ(device_x.values(), x);
            EXPECT_EQ(device_y.values(), y);
        }

        /// @brief Tests that the SYCL copy reads every other element of x into y.
        TEST_F(Medium_SYCLLevel1, CopyReadsEveryOtherElement)
        {
            const std::vector<float> x = operand(3);
            SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), static_cast<std::size_t>(K_LENGTH / 2));
            std::vector<float> expected(static_cast<std::size_t>(K_LENGTH / 2));

            Kernels::SYCL::copy<float>(queue(), K_LENGTH / 2, device_x.get(), 2, device_y.get(),
                                       1, {}).wait();
            Ref::copy<float>(K_LENGTH / 2, x.data(), 2, expected.data(), 1);

            EXPECT_EQ(device_y.values(), expected);
        }

        /// @brief Tests that the SYCL swap exchanges every other element of x with y, on both vectors.
        TEST_F(Medium_SYCLLevel1, SwapExchangesEveryOtherElement)
        {
            const std::vector<float> x = operand(3);
            const std::vector<float> y = operand(4);
            const std::size_t half = static_cast<std::size_t>(K_LENGTH / 2);
            SharedArray<float> device_x(queue(), x);
            SharedArray<float> device_y(queue(), std::vector<float>(y.begin(), y.begin() + half));
            std::vector<float> expected_x = x;
            std::vector<float> expected_y(y.begin(), y.begin() + half);

            Kernels::SYCL::swap<float>(queue(), K_LENGTH / 2, device_x.get(), 2, device_y.get(),
                                       1, {}).wait();
            Ref::swap<float>(K_LENGTH / 2, expected_x.data(), 2, expected_y.data(), 1);

            EXPECT_EQ(device_x.values(), expected_x);
            EXPECT_EQ(device_y.values(), expected_y);
        }

        /// @brief Tests that the SYCL asum matches the reference exactly.
        TEST_F(Medium_SYCLLevel1, AsumMatchesTheReferenceExactly)
        {
            const std::vector<double> x = small_integers<double>(K_LENGTH, 4);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> result(queue(), 1);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);
            double expected = 0.0;

            Kernels::SYCL::asum<double>(queue(), K_LENGTH, device_x.get(), 1, result.get(),
                                        scratch.get(), K_SCRATCH_BYTES, {}).wait();
            Ref::asum<double>(K_LENGTH, x.data(), 1, &expected);

            EXPECT_EQ(result.get()[0], expected);
        }

        /// @brief Tests that the SYCL dot matches the reference exactly.
        TEST_F(Medium_SYCLLevel1, DotMatchesTheReferenceExactly)
        {
            const std::vector<double> x = small_integers<double>(K_LENGTH, 4);
            const std::vector<double> y = small_integers<double>(K_LENGTH, 5);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_y(queue(), y);
            SharedArray<double> result(queue(), 1);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);
            double expected = 0.0;

            Kernels::SYCL::dot<double>(queue(), K_LENGTH, device_x.get(), 1, device_y.get(), 1,
                                       result.get(), scratch.get(), K_SCRATCH_BYTES, {}).wait();
            Ref::dot<double>(K_LENGTH, x.data(), 1, y.data(), 1, &expected);

            EXPECT_EQ(result.get()[0], expected);
        }

        /// @brief Tests that the SYCL iamax returns the same index as the reference.
        TEST_F(Medium_SYCLLevel1, IamaxMatchesTheReference)
        {
            const std::vector<double> x = small_integers<double>(K_LENGTH, 4);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<int64_t> index(queue(), 1);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);
            int64_t first_largest = -1;

            Kernels::SYCL::iamax<double>(queue(), K_LENGTH, device_x.get(), 1, index.get(),
                                         scratch.get(), K_SCRATCH_BYTES, {}).wait();
            Ref::iamax<double>(K_LENGTH, x.data(), 1, &first_largest);

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

            // A staged sum of K_LENGTH squares rounds K_LENGTH times, so the norm is held to K_LENGTH * u * |norm|.
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
                norm.get()[0], expected, static_cast<std::size_t>(K_LENGTH))));
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
