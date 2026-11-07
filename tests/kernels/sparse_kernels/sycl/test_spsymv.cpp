/**************************************************************************/
/* test_spsymv.cpp                                                        */
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
 * @file test_spsymv.cpp
 * @brief Tests the scattered SYCL spsymv against the reference spsymv on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/sparse_kernels/csr_pattern.hpp>
#include <SushiBLAS/sparse_kernels/reference/spsymv.hpp>
#include <SushiBLAS/sparse_kernels/sycl/row_blocks.hpp>
#include <SushiBLAS/sparse_kernels/sycl/spsymv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include <SushiBLAS/sparse_fixtures/integer_csr.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The triangle spsymv reads and the row blocks it scatters over. */
            struct TriangleRow
            {
                    const char* name;    /**< The row suffix. */
                    bool upper;          /**< Whether the upper triangle is the one stored and read. */
                    std::size_t blocks;  /**< The row blocks of the scatter; 0 takes the count of the profile. */
            };

            /** @brief Returns each triangle at the block count of the profile, and the lower one at each block count. */
            std::vector<TriangleRow> triangle_rows()
            {
                return {{"Lower", false, 0},
                        {"Upper", true, 0},
                        {"LowerWithOneBlock", false, 1},
                        {"LowerWithTwoBlocks", false, 2},
                        {"LowerWithThreeBlocks", false, 3},
                        {"LowerWithSevenBlocks", false, 7},
                        {"LowerWithSixtyFourBlocks", false, 64},
                        {"LowerWithTheMaximumOfBlocks", false,
                         Kernels::SYCL::K_MAX_SCATTER_BLOCKS}};
            }

            /** @brief Returns the block count a row asks for, or the one of the profile when it asks for none. */
            std::size_t blocks_of(const TriangleRow& row, const Device::Profile& profile,
                                  int64_t outputs, int64_t nnz)
            {
                return row.blocks != 0 ? row.blocks
                                       : Kernels::SYCL::scatter_blocks(profile, outputs, nnz);
            }

            /// @brief Names a row, which gtest takes as the test's suffix.
            std::string triangle_row_name(const ::testing::TestParamInfo<TriangleRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs spsymv on the host CPU queue, one triangle at a time. */
        class Medium_SYCLSpsymv : public HostDeviceTest, public ::testing::WithParamInterface<TriangleRow>
        {

        };

        TEST_P(Medium_SYCLSpsymv, MatchesTheReferenceForEitherTriangle)
        {
            const bool upper = GetParam().upper;
            const int64_t n = 283;
            const IntegerCsr a = integer_csr(n, n, 4, true);
            const SharedArray<int64_t> offsets(queue(), a.offsets);
            const SharedArray<int64_t> columns(queue(), a.columns);
            const SharedArray<double> values(queue(), a.values);
            const Kernels::CsrPattern<int64_t> device_pattern{n, n, a.nnz(), 0, offsets.get(),
                                                              columns.get()};
            const Kernels::CsrPattern<int64_t> host_pattern{n, n, a.nnz(), 0, a.offsets.data(),
                                                            a.columns.data()};
            const std::size_t blocks =
                blocks_of(GetParam(), Device::profile_for(queue()), n, a.nnz());
            SharedArray<Kernels::Accumulator<double>> partials(
                queue(), blocks * static_cast<std::size_t>(n));
            const std::vector<double> x = small_integers<double>(n, 5);
            const SharedArray<double> device_x(queue(), x);
            std::vector<double> y = small_integers<double>(n, 6);
            SharedArray<double> device_y(queue(), y);

            Kernels::SYCL::spsymv(queue(), uplo_of(upper), device_pattern, values.get(), 1.0,
                                  device_x.get(), 1, 3.0, device_y.get(), 1, partials.get(),
                                  blocks, {}).wait();
            Kernels::Reference::spsymv(uplo_of(upper), host_pattern, a.values.data(), 1.0,
                                       x.data(), 1, 3.0, y.data(), 1);

            EXPECT_EQ(device_y.values(), y);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLSpsymv, ::testing::ValuesIn(triangle_rows()),
                                 triangle_row_name);
    } // namespace Test
} // namespace SushiBLAS
