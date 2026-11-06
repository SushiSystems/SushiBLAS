/**************************************************************************/
/* test_upsample.cpp                                                      */
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
 * @file test_upsample.cpp
 * @brief Tests the SYCL nearest upsample and its backward against the reference ones.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/upsample.hpp>
#include <SushiBLAS/spatial_kernels/sycl/upsample.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLUpsample : public HostDeviceTest
        {

        };

        namespace
        {
            /** @brief Returns the span of a 2 x 5 x 4 x 3 input upsampled by 2 and 3 to 10 x 12. */
            Spatial::WindowSpan upsample_span()
            {
                Spatial::WindowSpan span = nhwc_span(2, 5, 4, 3, Spatial::ConvGeometry{}, 0);
                span.out_h = 10;
                span.out_w = 12;
                span.out_row_stride = span.out_w * 3;
                span.out_batch_stride = span.out_h * span.out_w * 3;
                return span;
            }
        } // namespace

        TEST_F(Medium_SYCLUpsample, ForwardMatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = upsample_span();
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_y(queue(), outputs);
            std::vector<double> y(outputs);

            Kernels::SYCL::upsample_nearest(queue(), span, 2, 3, device_x.get(), device_y.get(),
                                            {}).wait();
            Kernels::Reference::upsample_nearest(span, 2, 3, x.data(), y.data());

            EXPECT_EQ(device_y.values(), y);
        }

        TEST_F(Medium_SYCLUpsample, BackwardMatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = upsample_span();
            const std::size_t inputs = static_cast<std::size_t>(span.input_elements());
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            const SharedArray<double> device_dy(queue(), dy);
            SharedArray<double> device_dx(queue(), inputs);
            std::vector<double> dx(inputs);

            Kernels::SYCL::upsample_nearest_backward(queue(), span, 2, 3, device_dy.get(),
                                                     device_dx.get(), {}).wait();
            Kernels::Reference::upsample_nearest_backward(span, 2, 3, dy.data(), dx.data());

            EXPECT_EQ(device_dx.values(), dx);
        }
    } // namespace Test
} // namespace SushiBLAS
