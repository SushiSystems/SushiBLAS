/**************************************************************************/
/* test_blend.cpp                                                         */
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

/**
 * @file test_blend.cpp
 * @brief Tests that blend.hpp alone reaches the reference and the SYCL blend kernels.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blend_kernels/blend.hpp>

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_BlendKernels, AReferenceKernelIsReachable)
        {
            const std::vector<float> points{1.0f, 1.0f, 1.0f};
            const std::vector<float> matrices{1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3};
            const std::vector<int32_t> indices{0};
            const std::vector<float> weights{1.0f};
            std::vector<float> out(3, 0.0f);

            Kernels::Reference::blend_transforms(1, 3, 3, 1, 1, points.data(), matrices.data(),
                                                 indices.data(), weights.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{2.0f, 3.0f, 4.0f}));
        }

        /** @brief Runs a kernel reached through blend.hpp on the host CPU queue. */
        class Medium_BlendKernelsOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_BlendKernelsOnDevice, ASYCLKernelIsReachable)
        {
            const SharedArray<float> points(queue(), std::vector<float>{1.0f, 1.0f, 1.0f});
            const SharedArray<float> matrices(
                queue(), std::vector<float>{1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3});
            const SharedArray<int32_t> indices(queue(), std::vector<int32_t>{0});
            const SharedArray<float> weights(queue(), std::vector<float>{1.0f});
            SharedArray<float> out(queue(), 3);

            Kernels::SYCL::blend_transforms(queue(), 1, 3, 3, 1, 1, points.get(), matrices.get(),
                                            indices.get(), weights.get(), out.get(), {}).wait();

            EXPECT_EQ(out.values(), (std::vector<float>{2.0f, 3.0f, 4.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
