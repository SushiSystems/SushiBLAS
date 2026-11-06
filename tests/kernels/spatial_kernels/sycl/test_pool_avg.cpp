/**************************************************************************/
/* test_pool_avg.cpp                                                      */
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
 * @file test_pool_avg.cpp
 * @brief Tests the SYCL average pool and its backward against the reference ones.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_avg.hpp>
#include <SushiBLAS/spatial_kernels/sycl/pool_avg.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/tensor_expectations/tolerance_bounds.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the rounded operations of one backward element: six divisions and five additions, rounded up.
            constexpr std::size_t K_BACKWARD_ROUNDINGS = 12;

            /// @brief Holds the largest magnitude on that path: the sum of six quotients of dy / 6 with |dy| at most 3.
            constexpr double K_BACKWARD_MAGNITUDE = 3.0;

            /** @brief Expects every element of got within tolerance of the one of want. */
            void expect_values_near(const std::vector<double>& got, const std::vector<double>& want,
                                    double tolerance)
            {
                ASSERT_EQ(got.size(), want.size());
                for (std::size_t i = 0; i < got.size(); ++i)
                    EXPECT_NEAR(got[i], want[i], tolerance) << "element " << i;
            }
        } // namespace

        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLPoolAvg : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLPoolAvg, ForwardEqualsTheReferenceBitForBit)
        {
            // Each window sum is an exact integer and both kernels divide it by the same area, so the quotients agree.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_y(queue(), outputs);
            std::vector<double> y(outputs);

            Kernels::SYCL::pool_avg(queue(), span, device_x.get(), device_y.get(), {}).wait();
            Kernels::Reference::pool_avg(span, x.data(), y.data());

            EXPECT_EQ(device_y.values(), y);
        }

        TEST_F(Medium_SYCLPoolAvg, BackwardMatchesTheReferenceWithinItsRoundings)
        {
            // Each element sums six quotients dy / 6, which round, so the kernels agree within a bound.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t inputs = static_cast<std::size_t>(span.input_elements());
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            const SharedArray<double> device_dy(queue(), dy);
            SharedArray<double> device_dx(queue(), inputs);
            std::vector<double> dx(inputs);

            Kernels::SYCL::pool_avg_backward(queue(), span, device_dy.get(), device_dx.get(),
                                             {}).wait();
            Kernels::Reference::pool_avg_backward(span, dy.data(), dx.data());

            expect_values_near(device_dx.values(), dx, absolute_tolerance<double>(K_BACKWARD_ROUNDINGS, K_BACKWARD_MAGNITUDE));
        }
    } // namespace Test
} // namespace SushiBLAS
