/**************************************************************************/
/* test_window_dot_weights.cpp                                            */
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
 * @file test_window_dot_weights.cpp
 * @brief Tests the SYCL weight gradient in both launch shapes against the reference.
 * @author Mustafa Garip
 */

#include <string>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot_weights.hpp>
#include <SushiBLAS/spatial_kernels/sycl/window_dot_weights.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief One launch shape of the weight gradient. */
            struct LaunchRow
            {
                    const char* name;   /**< The row's suffix. */
                    std::size_t local;  /**< The work-group size of the reduction. */
            };

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string launch_row_name(const ::testing::TestParamInfo<LaunchRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the kernel on the host CPU queue, one launch shape at a time. */
        class Medium_SYCLWindowDotWeights : public HostDeviceTest,
                                            public ::testing::WithParamInterface<LaunchRow>
        {

        };

        TEST_P(Medium_SYCLWindowDotWeights, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const Spatial::TapSpan taps{span.geometry.kernel_w * 3, 3, 1};
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_dy(queue(), dy);
            std::vector<double> dw(static_cast<std::size_t>(span.patch_elements()));
            Kernels::Reference::window_dot_weights_backward(span, taps, dy.data(), x.data(),
                                                            dw.data());
            SharedArray<double> device_dw(queue(), static_cast<std::size_t>(span.patch_elements()));

            Kernels::SYCL::window_dot_weights_backward(queue(), span, taps,
                                                       Reduce::AxisPlan{GetParam().local},
                                                       device_dy.get(), device_x.get(),
                                                       device_dw.get(), {}).wait();

            EXPECT_EQ(device_dw.values(), dw);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLWindowDotWeights,
                                 ::testing::Values(LaunchRow{"LocalOne", 1}, LaunchRow{"LocalSixteen", 16}),
                                 launch_row_name);
    } // namespace Test
} // namespace SushiBLAS
