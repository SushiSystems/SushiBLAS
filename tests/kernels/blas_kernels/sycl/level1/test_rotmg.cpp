/**************************************************************************/
/* test_rotmg.cpp                                                         */
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
 * @file test_rotmg.cpp
 * @brief Tests the SYCL rotmg against the reference rotmg on the host device.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotmg.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotmg.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs rotmg on the host CPU queue. */
        class Medium_SYCLRotmg : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLRotmg, MatchesTheReferenceBitForBit)
        {
            const std::vector<std::vector<double>> inputs{
                {2.0, 1.0, 1.5, 0.5}, {1.0, 3.0, 1.0, 4.0}, {2.0, 1.0, 1.5, 0.0},
                {1.0, -1.0, 2.0, 3.0}, {1e-10, 1e10, 1.0, 1.0}};

            for (const std::vector<double>& in : inputs)
            {
                SCOPED_TRACE(in[3]);
                SharedArray<double> state(queue(), std::vector<double>{in[0], in[1], in[2], in[3]});
                SharedArray<double> device_param(queue(), 5);
                double* s = state.get();
                Kernels::SYCL::rotmg(queue(), s, s + 1, s + 2, s + 3, device_param.get(),
                                     {}).wait();

                double d1 = in[0];
                double d2 = in[1];
                double x1 = in[2];
                std::vector<double> param(5, 0.0);
                Kernels::Reference::rotmg(&d1, &d2, &x1, &in[3], param.data());

                EXPECT_EQ(state.values(), (std::vector<double>{d1, d2, x1, in[3]}));
                EXPECT_EQ(device_param.values(), param);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
