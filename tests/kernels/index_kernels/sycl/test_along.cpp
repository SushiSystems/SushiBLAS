/**************************************************************************/
/* test_along.cpp                                                         */
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
 * @file test_along.cpp
 * @brief Tests the SYCL take_along and put_along against the reference ones.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/index_kernels/reference/along.hpp>
#include <SushiBLAS/index_kernels/sycl/along.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLAlong : public HostDeviceTest
        {

        };

        namespace
        {
            /** @brief Holds the extents of a 6 x 4 matrix and one index per row, two of them outside it. */
            struct AlongCase
            {
                Kernels::MatrixSpan values{6, 4, 4, 1};
                std::vector<int64_t> index{3, 0, -1, 2, 4, 1};
            };
        } // namespace

        TEST_F(Medium_SYCLAlong, TakeAlongMatchesTheReference)
        {
            const AlongCase shape;
            const std::vector<double> values = small_integers<double>(24, 1);
            const SharedArray<double> device_values(queue(), values);
            const SharedArray<int64_t> device_index(queue(), shape.index);
            SharedArray<double> device_picked(queue(), static_cast<std::size_t>(6));
            std::vector<double> picked(6);

            Kernels::SYCL::take_along<double, int64_t>(queue(), shape.values, device_values.get(),
                                                       device_index.get(), 1,
                                                       device_picked.get(), 1, {}).wait();
            Kernels::Reference::take_along<double, int64_t>(shape.values, values.data(),
                                                            shape.index.data(), 1, picked.data(),
                                                            1);

            EXPECT_EQ(device_picked.values(), picked);
        }

        TEST_F(Medium_SYCLAlong, PutAlongMatchesTheReference)
        {
            const AlongCase shape;
            const std::vector<double> picked = small_integers<double>(6, 2);
            const SharedArray<double> device_picked(queue(), picked);
            const SharedArray<int64_t> device_index(queue(), shape.index);
            SharedArray<double> device_values(queue(), static_cast<std::size_t>(24));
            std::vector<double> values(24);

            Kernels::SYCL::put_along<double, int64_t>(queue(), device_picked.get(), 1,
                                                      device_index.get(), 1, shape.values,
                                                      device_values.get(), {}).wait();
            Kernels::Reference::put_along<double, int64_t>(picked.data(), 1, shape.index.data(),
                                                           1, shape.values, values.data());

            EXPECT_EQ(device_values.values(), values);
        }
    } // namespace Test
} // namespace SushiBLAS
