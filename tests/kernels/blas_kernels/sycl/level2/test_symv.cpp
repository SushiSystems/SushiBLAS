/**************************************************************************/
/* test_symv.cpp                                                          */
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
 * @file test_symv.cpp
 * @brief Tests the SYCL symv against the reference symv on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/symv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/symv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Names a row after its layout and its triangle.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool>>& info)
        {
            return layout_uplo_label(std::get<0>(info.param), std::get<1>(info.param));
        }

        /// @brief Runs symv on the host CPU queue, one layout and triangle at a time.
        class Medium_SYCLSymv : public HostDeviceTest, public ::testing::WithParamInterface<std::tuple<bool, bool>>
        {

        };

        /// @brief Tests that the SYCL symv matches the reference for one layout and triangle.
        TEST_P(Medium_SYCLSymv, MatchesTheReference)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const int64_t n = 45;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> x = small_integers<double>(n, 2);
            const SharedArray<double> device_a(queue(), a);
            const SharedArray<double> device_x(queue(), x);
            std::vector<double> y = small_integers<double>(n, 3);
            SharedArray<double> device_y(queue(), y);

            Kernels::SYCL::symv(queue(), layout, uplo_of(upper), n, 2.0, device_a.get(),
                                n, device_x.get(), 1, 3.0, device_y.get(), 1, {}).wait();
            Kernels::Reference::symv(layout, uplo_of(upper), n, 2.0, a.data(), n,
                                     x.data(), 1, 3.0, y.data(), 1);

            EXPECT_EQ(device_y.values(), y);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsAndTriangles, Medium_SYCLSymv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool()), row_name);
    } // namespace Test
} // namespace SushiBLAS
