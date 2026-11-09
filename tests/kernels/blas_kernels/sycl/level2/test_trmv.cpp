/**************************************************************************/
/* test_trmv.cpp                                                          */
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
 * @file test_trmv.cpp
 * @brief Tests the SYCL trmv against the reference trmv for every flag on the host device.
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
#include <SushiBLAS/blas_kernels/reference/level2/trmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/trmv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Names a row after its layout, triangle, transpose and diagonal.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool, bool, bool>>& info)
        {
            return triangular_label(std::get<0>(info.param), std::get<1>(info.param),
                                    std::get<2>(info.param), std::get<3>(info.param));
        }

        /// @brief Runs trmv on the host CPU queue, one layout, triangle, transpose and diagonal at a time.
        class Medium_SYCLTrmv : public HostDeviceTest, public ::testing::WithParamInterface<std::tuple<bool, bool, bool, bool>>
        {

        };

        /// @brief Tests that the SYCL trmv matches the reference for one layout, triangle, transpose and diagonal.
        TEST_P(Medium_SYCLTrmv, MatchesTheReference)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const bool transposed = std::get<2>(GetParam());
            const bool unit = std::get<3>(GetParam());
            const int64_t n = 47;
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const SharedArray<double> device_a(queue(), a);
            SharedArray<double> work(queue(), static_cast<std::size_t>(n));
            std::vector<double> x = small_integers<double>(2 * n, 2);
            SharedArray<double> device_x(queue(), x);

            Kernels::SYCL::trmv(queue(), layout, uplo_of(upper),
                                transpose_of(transposed), diag_of(unit), n,
                                device_a.get(), n, device_x.get(), 2, work.get(),
                                {}).wait();
            Kernels::Reference::trmv(layout, uplo_of(upper),
                                     transpose_of(transposed), diag_of(unit), n,
                                     a.data(), n, x.data(), 2);

            EXPECT_EQ(device_x.values(), x);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsTrianglesTransposesAndDiagonals, Medium_SYCLTrmv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool(), ::testing::Bool()),
                                 row_name);
    } // namespace Test
} // namespace SushiBLAS
