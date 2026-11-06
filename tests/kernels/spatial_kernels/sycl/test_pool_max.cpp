/**************************************************************************/
/* test_pool_max.cpp                                                      */
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
 * @file test_pool_max.cpp
 * @brief Tests the SYCL max pool and its backward against the reference ones.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_max.hpp>
#include <SushiBLAS/spatial_kernels/sycl/pool_max.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLPoolMax : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLPoolMax, ForwardValuesMatchTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_y(queue(), outputs);
            SharedArray<int64_t> device_index(queue(), outputs);
            std::vector<double> y(outputs);
            std::vector<int64_t> index(outputs);

            Kernels::SYCL::pool_max(queue(), span, device_x.get(), device_y.get(),
                                    device_index.get(), {}).wait();
            Kernels::Reference::pool_max(span, x.data(), y.data(), index.data());

            EXPECT_EQ(device_y.values(), y);
        }

        TEST_F(Medium_SYCLPoolMax, RecordedIndicesMatchTheReference)
        {
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_y(queue(), outputs);
            SharedArray<int64_t> device_index(queue(), outputs);
            std::vector<double> y(outputs);
            std::vector<int64_t> index(outputs);

            Kernels::SYCL::pool_max(queue(), span, device_x.get(), device_y.get(),
                                    device_index.get(), {}).wait();
            Kernels::Reference::pool_max(span, x.data(), y.data(), index.data());

            EXPECT_EQ(device_index.values(), index);
        }

        TEST_F(Medium_SYCLPoolMax, BackwardMatchesTheReferenceGivenTheSameIndices)
        {
            // The reference indices feed both backward passes, so only the backward is compared.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::size_t inputs = static_cast<std::size_t>(span.input_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            std::vector<double> y(outputs);
            std::vector<int64_t> index(outputs);
            Kernels::Reference::pool_max(span, x.data(), y.data(), index.data());
            const SharedArray<double> device_dy(queue(), dy);
            const SharedArray<int64_t> device_index(queue(), index);
            SharedArray<double> device_dx(queue(), inputs);
            std::vector<double> dx(inputs);

            Kernels::SYCL::pool_max_backward(queue(), span, device_dy.get(), device_index.get(),
                                             device_dx.get(), {}).wait();
            Kernels::Reference::pool_max_backward(span, dy.data(), index.data(), dx.data());

            EXPECT_EQ(device_dx.values(), dx);
        }
    } // namespace Test
} // namespace SushiBLAS
