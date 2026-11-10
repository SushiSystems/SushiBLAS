/**************************************************************************/
/* test_syrk.cpp                                                          */
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
 * @file test_syrk.cpp
 * @brief Tests the tiled and the packed SYCL syrk against the reference syrk on the host device, the beta zero and short scratch cases, and the tile check.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_kernels/reference/level3/syrk.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/syrk.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The order of C; odd, so no tile divides it.
        constexpr int64_t K_ORDER = 67;

        /// @brief The inner dimension of the update.
        constexpr int64_t K_DEPTH = 45;

        /// @brief The scratch the packed path may take; none sends the call down the tiled path.
        constexpr std::size_t K_SCRATCH_BYTES = std::size_t{8} << 20;

        /// @brief Names a row after its path, layout, triangle and transpose.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<bool, bool, bool, bool>>& info)
        {
            return std::string(std::get<0>(info.param) ? "Packed" : "Tiled") +
                   layout_uplo_label(std::get<1>(info.param), std::get<2>(info.param)) +
                   (std::get<3>(info.param) ? "Trans" : "NoTrans");
        }

        /// @brief Runs syrk on the host CPU queue, one path, layout, triangle and transpose at a time.
        class Medium_SYCLSyrk : public HostDeviceTest, public ::testing::WithParamInterface<std::tuple<bool, bool, bool, bool>>
        {

        };

        /// @brief Tests that the SYCL syrk matches the reference on the path, layout, triangle and transpose of a row.
        TEST_P(Medium_SYCLSyrk, MatchesTheReference)
        {
            const bool packed_path = std::get<0>(GetParam());
            const Core::Layout layout = layout_of(std::get<1>(GetParam()));
            const bool upper = std::get<2>(GetParam());
            const bool transposed = std::get<3>(GetParam());
            // Every operand is a small integer, so both kernels compute without rounding.
            const std::vector<double> a = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const SharedArray<double> device_a(queue(), a);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);
            const std::size_t scratch_bytes = packed_path ? K_SCRATCH_BYTES : std::size_t{0};
            const Kernels::PackedProductPlan plan = Kernels::plan_packed_product<double>(
                Device::profile_for(queue()), K_ORDER, K_ORDER, K_DEPTH, 1);
            if (packed_path && !Kernels::packed_product_is_ready(plan, scratch.get(), scratch_bytes))
                GTEST_SKIP() << "the profile of the host queue does not select the packed kernel";
            const bool by_rows = (layout == Core::Layout::RowMajor) != transposed;
            const int64_t lda = by_rows ? K_DEPTH : K_ORDER;
            std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 2);
            SharedArray<double> device_c(queue(), c);

            Kernels::SYCL::syrk(queue(), layout, uplo_of(upper), transpose_of(transposed),
                                K_ORDER, K_DEPTH, 2.0, device_a.get(), lda, 0, -1.0,
                                device_c.get(), K_ORDER, 0, 1,
                                packed_path ? scratch.get() : nullptr, scratch_bytes, {}).wait();
            Kernels::Reference::syrk(layout, uplo_of(upper), transpose_of(transposed), K_ORDER,
                                     K_DEPTH, 2.0, a.data(), lda, -1.0, c.data(), K_ORDER);

            EXPECT_EQ(device_c.values(), c);
        }

        INSTANTIATE_TEST_SUITE_P(PathsLayoutsTrianglesAndTransposes, Medium_SYCLSyrk,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool(), ::testing::Bool()),
                                 row_name);

        /// @brief Runs the tile check of the tiled syrk on a profile value, with no device.
        class Small_SyrkTileCheck : public ::testing::Test
        {
            protected:
                /** @brief Returns the profile of a device with the given local memory and the work-group size of 256. */
                [[nodiscard]] static Device::Profile profile_with_local_memory(std::size_t bytes)
                {
                    Device::Profile profile;
                    profile.local_mem_bytes = bytes;
                    profile.max_work_group_size = 256;
                    return profile;
                }
        };

        // The tile of double stages (32 * 16 + 16 * 32) * 8 = 8192 B in 64-item work-groups.
        TEST_F(Small_SyrkTileCheck, ADeviceWithTooLittleLocalMemoryIsRefusedWithARuntimeError)
        {
            EXPECT_THROW(Kernels::SYCL::require_syrk_tile<double>(profile_with_local_memory(8191)),
                         std::runtime_error);
        }

        /// @brief Runs syrk on the host CPU queue with a C that holds NaN, one path at a time; false is the tiled path.
        class Medium_SYCLSyrkBetaZero : public HostDeviceTest, public ::testing::WithParamInterface<bool>
        {
            protected:
                /** @brief Skips a packed row when the profile of the host queue does not select the packed kernel. */
                void SetUp() override
                {
                    HostDeviceTest::SetUp();
                    if (IsSkipped())
                        return;
                    const Kernels::PackedProductPlan plan = Kernels::plan_packed_product<double>(
                        Device::profile_for(queue()), K_ORDER, K_ORDER, K_DEPTH, 1);
                    if (GetParam() && !plan.selected)
                        GTEST_SKIP() << "the profile of the host queue does not select the packed kernel";
                }
        };

        /// @brief Tests that a beta of zero overwrites the triangle without reading C, so a NaN in C never reaches the result.
        TEST_P(Medium_SYCLSyrkBetaZero, ANanInCNeverReachesTheResult)
        {
            const bool packed_path = GetParam();
            const std::vector<double> a = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const SharedArray<double> device_a(queue(), a);
            SharedArray<std::byte> scratch(queue(), K_SCRATCH_BYTES);
            SharedArray<double> device_c(queue(), std::vector<double>(K_ORDER * K_ORDER,
                                                                       std::numeric_limits<double>::quiet_NaN()));
            std::vector<double> c(K_ORDER * K_ORDER, 0.0);

            Kernels::SYCL::syrk(queue(), Core::Layout::RowMajor, Core::Uplo::Upper,
                                Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 2.0, device_a.get(),
                                K_DEPTH, 0, 0.0, device_c.get(), K_ORDER, 0, 1,
                                packed_path ? scratch.get() : nullptr,
                                packed_path ? K_SCRATCH_BYTES : std::size_t{0}, {}).wait();
            Kernels::Reference::syrk(Core::Layout::RowMajor, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 2.0, a.data(),
                                     K_DEPTH, 0.0, c.data(), K_ORDER);

            EXPECT_EQ(triangle_of(device_c.values(), K_ORDER, true, false),
                      triangle_of(c, K_ORDER, true, false));
        }

        INSTANTIATE_TEST_SUITE_P(Paths, Medium_SYCLSyrkBetaZero, ::testing::Bool());

        /// @brief Runs syrk on the host CPU queue with a scratch one byte short of the packed plan.
        class Medium_SYCLSyrkShortScratch : public HostDeviceTest
        {
            protected:
                /** @brief Returns the packed plan of the update on the profile of the host queue. */
                [[nodiscard]] static Kernels::PackedProductPlan plan()
                {
                    return Kernels::plan_packed_product<double>(Device::profile_for(queue()), K_ORDER,
                                                                K_ORDER, K_DEPTH, 1);
                }

                /** @brief Skips the test when the profile of the host queue does not select the packed kernel. */
                void SetUp() override
                {
                    HostDeviceTest::SetUp();
                    if (IsSkipped())
                        return;
                    if (!plan().selected)
                        GTEST_SKIP() << "the profile of the host queue does not select the packed kernel";
                }
        };

        /// @brief Tests that a scratch below the bytes of the packed plan sends the call down the tiled path, which gives the reference result.
        TEST_F(Medium_SYCLSyrkShortScratch, AScratchBelowThePlanTakesTheTiledPath)
        {
            const std::vector<double> a = small_integers<double>(K_ORDER * K_DEPTH, 1);
            const SharedArray<double> device_a(queue(), a);
            SharedArray<std::byte> scratch(queue(), plan().scratch_bytes);
            std::vector<double> c = small_integers<double>(K_ORDER * K_ORDER, 2);
            SharedArray<double> device_c(queue(), c);

            Kernels::SYCL::syrk(queue(), Core::Layout::RowMajor, Core::Uplo::Upper,
                                Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 2.0, device_a.get(),
                                K_DEPTH, 0, -1.0, device_c.get(), K_ORDER, 0, 1, scratch.get(),
                                plan().scratch_bytes - 1, {}).wait();
            Kernels::Reference::syrk(Core::Layout::RowMajor, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, K_ORDER, K_DEPTH, 2.0, a.data(),
                                     K_DEPTH, -1.0, c.data(), K_ORDER);

            EXPECT_EQ(device_c.values(), c);
        }
    } // namespace Test
} // namespace SushiBLAS
