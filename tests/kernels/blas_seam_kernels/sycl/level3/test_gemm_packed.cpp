/**************************************************************************/
/* test_gemm_packed.cpp                                                   */
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
 * @file test_gemm_packed.cpp
 * @brief Tests the packed SYCL gemm under three blockings against the reference gemm.
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
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_packed.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/packed_product_harness/packed_product_harness.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Names the cache blockings the packed gemm runs under. */
        enum class BlockingName : std::uint8_t
        {
            Smallest,
            Ragged,
            Widest
        };

        /** @brief Holds one layout, one transpose of A and one blocking of a packed gemm. */
        struct PackedGemmCase
        {
            Core::Layout layout; /**< The storage order of the operands. */
            bool trans_a;        /**< Whether A is transposed. */
            BlockingName blocking; /**< The cache blocking the plan is built from. */
        };

        /** @brief Returns the blocking a BlockingName stands for. */
        static Device::CacheBlocking blocking_of(BlockingName name)
        {
            if (name == BlockingName::Smallest)
                return smallest_blocking();
            return (name == BlockingName::Ragged) ? ragged_blocking<double>() : widest_blocking();
        }

        /** @brief Returns both layouts crossed with the transposes of A and the three blockings. */
        static std::vector<PackedGemmCase> packed_gemm_cases()
        {
            std::vector<PackedGemmCase> cases;
            for (const Core::Layout layout : {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
                for (const bool trans_a : {false, true})
                    for (const BlockingName blocking :
                         {BlockingName::Smallest, BlockingName::Ragged, BlockingName::Widest})
                        cases.push_back({layout, trans_a, blocking});
            return cases;
        }

        /** @brief Names a case for the test name. */
        static std::string packed_gemm_case_name(const ::testing::TestParamInfo<PackedGemmCase>& info)
        {
            static const char* const blockings[] = {"Smallest", "Ragged", "Widest"};
            return std::string(info.param.layout == Core::Layout::RowMajor ? "RowMajor"
                                                                            : "ColumnMajor") +
                   (info.param.trans_a ? "TransA" : "PlainA") +
                   blockings[static_cast<std::size_t>(info.param.blocking)];
        }

        /** @brief Runs the packed gemm on the host CPU queue, with one case at a time. */
        class Medium_SYCLGemmPacked : public HostDeviceTest,
                                      public ::testing::WithParamInterface<PackedGemmCase>
        {

        };

        TEST_P(Medium_SYCLGemmPacked, EveryBlockingMatchesTheReference)
        {
            // The patterned operands are small integers, so every blocking computes exactly.
            const PackedGemmCase row = GetParam();
            GemmProblem<double> problem(queue(), row.layout, transpose_of(row.trans_a),
                                        Core::Transpose::NoTrans, 45, 38, 52, 2);
            Kernels::PackedProductPlan plan;

            const std::vector<double> packed =
                problem.packed(blocking_of(row.blocking), 4, 3.0, 1.0, &plan);

            ASSERT_GT(plan.blocks, 0);
            EXPECT_EQ(packed, problem.expected(3.0, 1.0));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLGemmPacked, ::testing::ValuesIn(packed_gemm_cases()),
                                 packed_gemm_case_name);

        /** @brief Runs the packed gemm entry point on the host CPU queue. */
        class Medium_SYCLGemmPackedEntry : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLGemmPackedEntry, TheEntryPointWithScratchMatchesTheReference)
        {
            const int64_t m = 70;
            const int64_t n = 66;
            const int64_t k = 58;
            const std::vector<float> a = small_integers<float>(m * k, 1);
            const std::vector<float> b = small_integers<float>(k * n, 2);
            std::vector<float> c = small_integers<float>(m * n, 3);
            const SharedArray<float> device_a(queue(), a);
            const SharedArray<float> device_b(queue(), b);
            SharedArray<float> device_c(queue(), c);
            const std::size_t scratch_bytes = std::size_t{16} << 20;
            SharedArray<std::byte> scratch(queue(), scratch_bytes);

            Kernels::SYCL::gemm<float>(queue(), Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                       Core::Transpose::NoTrans, m, n, k, 1.0f, device_a.get(),
                                       k, 0, device_b.get(), n, 0, 2.0f, device_c.get(), n, 0, 1,
                                       scratch.get(), scratch_bytes, {}).wait();
            Kernels::Reference::gemm<float>(Core::Layout::RowMajor, Core::Transpose::NoTrans,
                                            Core::Transpose::NoTrans, m, n, k, 1.0f, a.data(), k,
                                            b.data(), n, 2.0f, c.data(), n);

            EXPECT_EQ(device_c.values(), c);
        }
    } // namespace Test
} // namespace SushiBLAS
