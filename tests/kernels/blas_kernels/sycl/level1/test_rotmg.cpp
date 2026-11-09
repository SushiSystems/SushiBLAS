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

#include <string>
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
        /// @brief One input of rotmg (d1, d2, x1, y1) and the name of its row.
        struct InputRow
        {
            const char* name;
            double d1;
            double d2;
            double x1;
            double y1;
        };

        /// @brief Names a row after its input.
        static std::string input_name(const ::testing::TestParamInfo<InputRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs rotmg on the host CPU queue, one input at a time.
        class Medium_SYCLRotmg : public HostDeviceTest, public ::testing::WithParamInterface<InputRow>
        {

        };

        /// @brief Tests that the SYCL rotmg matches the reference bit for bit.
        TEST_P(Medium_SYCLRotmg, MatchesTheReferenceBitForBit)
        {
            const InputRow& in = GetParam();
            SharedArray<double> state(queue(), std::vector<double>{in.d1, in.d2, in.x1, in.y1});
            SharedArray<double> device_param(queue(), 5);
            double* s = state.get();
            double d1 = in.d1;
            double d2 = in.d2;
            double x1 = in.x1;
            const double y1 = in.y1;
            std::vector<double> param(5, 0.0);

            Kernels::SYCL::rotmg(queue(), s, s + 1, s + 2, s + 3, device_param.get(), {}).wait();
            Kernels::Reference::rotmg(&d1, &d2, &x1, &y1, param.data());

            EXPECT_EQ(state.values(), (std::vector<double>{d1, d2, x1, y1}));
            EXPECT_EQ(device_param.values(), param);
        }

        INSTANTIATE_TEST_SUITE_P(Inputs, Medium_SYCLRotmg,
                                 ::testing::Values(InputRow{"Positive", 2.0, 1.0, 1.5, 0.5},
                                                   InputRow{"LargeSecond", 1.0, 3.0, 1.0, 4.0},
                                                   InputRow{"ZeroSecond", 2.0, 1.0, 1.5, 0.0},
                                                   InputRow{"NegativeFirst", 1.0, -1.0, 2.0, 3.0},
                                                   InputRow{"Extremes", 1e-10, 1e10, 1.0, 1.0}),
                                 input_name);
    } // namespace Test
} // namespace SushiBLAS
