/**************************************************************************/
/* test_spatial.cpp                                                       */
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
 * @file test_spatial.cpp
 * @brief Tests that spatial.hpp alone reaches the reference and the SYCL spatial kernels.
 * @author Mustafa Garip
 */

#include <SushiBLAS/spatial_kernels/spatial.hpp>

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include "nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_SpatialKernels, AReferenceKernelIsReachable)
        {
            const Spatial::WindowSpan span = nhwc_span(1, 1, 2, 1, Spatial::ConvGeometry{}, 0);
            const std::vector<float> in{4.0f, 6.0f};
            std::vector<float> out(2, 0.0f);

            Kernels::Reference::pool_avg(span, in.data(), out.data());

            EXPECT_EQ(out, in);
        }

        /** @brief Runs a kernel reached through spatial.hpp on the host CPU queue. */
        class Medium_SpatialKernelsOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_SpatialKernelsOnDevice, ASYCLKernelIsReachable)
        {
            const Spatial::WindowSpan span = nhwc_span(1, 1, 2, 1, Spatial::ConvGeometry{}, 0);
            const SharedArray<float> in(queue(), std::vector<float>{4.0f, 6.0f});
            SharedArray<float> out(queue(), 2);

            Kernels::SYCL::pool_avg(queue(), span, in.get(), out.get(), {}).wait();

            EXPECT_EQ(out.values(), (std::vector<float>{4.0f, 6.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
