/**************************************************************************/
/* test_ger.cpp                                                           */
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
 * @file test_ger.cpp
 * @brief Tests the SYCL ger against the reference ger on the host device.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/ger.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/ger.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Names a row after its layout.
        static std::string layout_name(const ::testing::TestParamInfo<bool>& info)
        {
            return info.param ? "RowMajor" : "ColumnMajor";
        }

        /// @brief Runs ger on the host CPU queue, one layout at a time.
        class Medium_SYCLGer : public HostDeviceTest, public ::testing::WithParamInterface<bool>
        {

        };

        /// @brief Tests that the SYCL ger matches the reference in one layout.
        TEST_P(Medium_SYCLGer, MatchesTheReferenceInEitherLayout)
        {
            const Core::Layout layout = layout_of(GetParam());
            const int64_t m = 33;
            const int64_t n = 47;
            const std::vector<float> x = small_integers<float>(m, 2);
            const std::vector<float> y = small_integers<float>(2 * n, 3);
            const SharedArray<float> device_x(queue(), x);
            const SharedArray<float> device_y(queue(), y);
            const int64_t lda = (layout == Core::Layout::RowMajor) ? n : m;
            std::vector<float> a = small_integers<float>(m * n, 1);
            SharedArray<float> device_a(queue(), a);

            Kernels::SYCL::ger(queue(), layout, m, n, 3.0f, device_x.get(), 1, device_y.get(), 2,
                               device_a.get(), lda, {}).wait();
            Kernels::Reference::ger(layout, m, n, 3.0f, x.data(), 1, y.data(), 2, a.data(), lda);

            EXPECT_EQ(device_a.values(), a);
        }

        INSTANTIATE_TEST_SUITE_P(Layouts, Medium_SYCLGer, ::testing::Bool(), layout_name);
    } // namespace Test
} // namespace SushiBLAS
