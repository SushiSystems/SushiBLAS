/**************************************************************************/
/* test_rotg.cpp                                                          */
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
 * @file test_rotg.cpp
 * @brief Tests the SYCL rotg against the reference rotg on the host device.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotg.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotg.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief One input pair of rotg and the name of its row.
            struct PairRow
            {
                const char* name;
                double a;
                double b;
            };
        } // namespace

        /// @brief Names a row after its input pair.
        static std::string pair_name(const ::testing::TestParamInfo<PairRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs rotg on the host CPU queue, one input pair at a time.
        class Medium_SYCLRotg : public HostDeviceTest, public ::testing::WithParamInterface<PairRow>
        {

        };

        /// @brief Tests that the SYCL rotg matches the reference bit for bit.
        TEST_P(Medium_SYCLRotg, MatchesTheReferenceBitForBit)
        {
            SharedArray<double> values(queue(), std::vector<double>{GetParam().a, GetParam().b, 0, 0});
            double* v = values.get();
            double a = GetParam().a;
            double b = GetParam().b;
            double c = 0.0;
            double s = 0.0;

            Kernels::SYCL::rotg(queue(), v, v + 1, v + 2, v + 3, {}).wait();
            Kernels::Reference::rotg(&a, &b, &c, &s);

            EXPECT_EQ(values.values(), (std::vector<double>{a, b, c, s}));
        }

        INSTANTIATE_TEST_SUITE_P(Pairs, Medium_SYCLRotg,
                                 ::testing::Values(PairRow{"ThreeFour", 3.0, 4.0},
                                                   PairRow{"MinusFourThree", -4.0, 3.0},
                                                   PairRow{"ZeroFirst", 0.0, 2.0},
                                                   PairRow{"ZeroSecond", 5.0, 0.0},
                                                   PairRow{"Tiny", 1e-200, 3e-200}),
                                 pair_name);
    } // namespace Test
} // namespace SushiBLAS
