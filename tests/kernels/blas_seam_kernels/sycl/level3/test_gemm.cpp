/**************************************************************************/
/* test_gemm.cpp                                                          */
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
 * @file test_gemm.cpp
 * @brief Tests the unpacked SYCL gemm against the reference gemm on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include <SushiLog/log/log_capture.hpp>
#include <SushiBLAS/test_runner/captured_record_count.hpp>
#include <SushiBLAS/test_runner/scoped_log_level.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds one layout and the two transposes of an unpacked gemm. */
        struct UnpackedGemmCase
        {
            Core::Layout layout; /**< The storage order of the operands. */
            bool trans_a;        /**< Whether A is transposed. */
            bool trans_b;        /**< Whether B is transposed. */
        };

        /** @brief Returns both layouts crossed with the four transpose pairs. */
        static std::vector<UnpackedGemmCase> unpacked_gemm_cases()
        {
            std::vector<UnpackedGemmCase> cases;
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
                for (const bool trans_a : {false, true})
                    for (const bool trans_b : {false, true})
                        cases.push_back({layout, trans_a, trans_b});
            return cases;
        }

        /** @brief Names a case for the test name. */
        static std::string unpacked_gemm_case_name(
            const ::testing::TestParamInfo<UnpackedGemmCase>& info)
        {
            return std::string(info.param.layout == Core::Layout::RowMajor ? "RowMajor"
                                                                            : "ColumnMajor") +
                   (info.param.trans_a ? "TransA" : "PlainA") +
                   (info.param.trans_b ? "TransB" : "PlainB");
        }

        /** @brief Runs the unpacked gemm on the host CPU queue. */
        class Medium_SYCLGemm : public HostDeviceTest
        {

        };

        /** @brief Runs the unpacked gemm on the host CPU queue, one case at a time. */
        class Medium_SYCLGemmCases : public HostDeviceTest,
                                     public ::testing::WithParamInterface<UnpackedGemmCase>
        {

        };

        TEST_P(Medium_SYCLGemmCases, MatchesTheReferenceInEveryLayoutTransposeAndBatch)
        {
            // The patterned operands are small integers, so both kernels compute exactly.
            const UnpackedGemmCase row = GetParam();
            GemmProblem<double> problem(queue(), row.layout, transpose_of(row.trans_a),
                                        transpose_of(row.trans_b), 37, 29, 41, 2);

            EXPECT_EQ(problem.unpacked(2.0, -1.0), problem.expected(2.0, -1.0));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLGemmCases, ::testing::ValuesIn(unpacked_gemm_cases()),
                                 unpacked_gemm_case_name);

        TEST_F(Medium_SYCLGemm, FloatAndAZeroBetaMatchTheReference)
        {
            GemmProblem<float> problem(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, 64, 48, 33, 1);
            problem.poison_output();
            const std::vector<float> got = problem.unpacked(1.0f, 0.0f);
            const std::vector<float> want = problem.expected(1.0f, 0.0f);

            // The padding keeps its NaN poison, and NaN never compares equal, so compare bits.
            ASSERT_EQ(got.size(), want.size());
            EXPECT_EQ(count_bit_mismatches(got.data(), want.data(), got.size()), std::size_t{0});
        }

        TEST(Medium_GemmFallbackLog, WarnsOncePerDeviceAndTile)
        {
            const Device::Profile first_device;
            const Device::Profile second_device;
            const SushiLog::LogCapture capture;

            Kernels::SYCL::Detail::warn_no_local_fallback(first_device, 65536, 256);
            Kernels::SYCL::Detail::warn_no_local_fallback(first_device, 65536, 256);
            Kernels::SYCL::Detail::warn_no_local_fallback(first_device, 65536, 256);

            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Warn,
                                    "falling back to the no-local variant"),
                      std::size_t{1});

            Kernels::SYCL::Detail::warn_no_local_fallback(second_device, 65536, 256);
            Kernels::SYCL::Detail::warn_no_local_fallback(first_device, 131072, 256);

            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Warn,
                                    "falling back to the no-local variant"),
                      std::size_t{3});
        }

        TEST(Medium_GemmFallbackLog, AnAppliedPinIsStatedOncePerProcess)
        {
            Kernels::GemmTileChoice choice;
            choice.pin = Kernels::GemmTilePinStatus::Applied;
            choice.role = Kernels::GemmTileRole::Balanced;
            const ScopedLogLevel level(SushiLog::LogLevel::Info);
            const SushiLog::LogCapture capture;

            Kernels::SYCL::Detail::state_applied_gemm_tile_pin(choice, 64, 64, 64);
            Kernels::SYCL::Detail::state_applied_gemm_tile_pin(choice, 128, 128, 128);

            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Info, "selects the"),
                      std::size_t{1});
        }

        TEST(Medium_GemmFallbackLog, APinThatWasNotAppliedStatesNothing)
        {
            Kernels::GemmTileChoice choice;
            choice.pin = Kernels::GemmTilePinStatus::Unknown;
            const ScopedLogLevel level(SushiLog::LogLevel::Info);
            const SushiLog::LogCapture capture;

            Kernels::SYCL::Detail::state_applied_gemm_tile_pin(choice, 64, 64, 64);

            EXPECT_EQ(count_records(capture.records(), SushiLog::LogLevel::Info, "selects the"),
                      std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
