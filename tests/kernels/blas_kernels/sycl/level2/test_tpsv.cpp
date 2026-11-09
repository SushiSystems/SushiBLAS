/**************************************************************************/
/* test_tpsv.cpp                                                          */
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
 * @file test_tpsv.cpp
 * @brief Tests the SYCL tpsv against the reference tpsv for every flag on the host device.
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
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/blas_kernels/reference/level2/tpsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tpsv.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>
#include <SushiBLAS/blas_test_oracle/blas_structured.hpp>
#include <SushiBLAS/blas_test_oracle/dense_oracle.hpp>
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

        /// @brief Runs tpsv on the host CPU queue, one layout, triangle, transpose and diagonal at a time.
        class Medium_SYCLTpsv : public HostDeviceTest, public ::testing::WithParamInterface<std::tuple<bool, bool, bool, bool>>
        {

        };

        /// @brief Tests that the SYCL tpsv matches the reference for one layout, triangle, transpose and diagonal.
        TEST_P(Medium_SYCLTpsv, MatchesTheReference)
        {
            const Core::Layout layout = layout_of(std::get<0>(GetParam()));
            const bool upper = std::get<1>(GetParam());
            const bool transposed = std::get<2>(GetParam());
            const bool unit = std::get<3>(GetParam());
            const int64_t n = 40;
            // A unit diagonal keeps every substitution exact in integers.
            const std::vector<double> a = unit_diagonal_matrix<double>(n, 1);
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            const std::vector<double> packed = to_packed(a, n, uplo_of(upper), layout);
            const SharedArray<double> device_packed(queue(), packed);
            // A small solution keeps every substitution step exact.
            const std::vector<double> solution = small_integers<double>(n, 2);
            std::vector<double> x =
                dense_matvec(triangle_of(a, n, upper, false), n, n, transposed,
                             1.0, solution, 0.0, solution);
            SharedArray<double> device_x(queue(), x);

            Kernels::SYCL::tpsv(queue(), layout, uplo_of(upper),
                                transpose_of(transposed), diag_of(unit), n,
                                device_packed.get(), device_x.get(), 1,
                                work.get(), {}).wait();
            Kernels::Reference::tpsv(layout, uplo_of(upper),
                                     transpose_of(transposed), diag_of(unit), n,
                                     packed.data(), x.data(), 1);

            EXPECT_EQ(device_x.values(), x);
            EXPECT_EQ(x, solution);
        }

        INSTANTIATE_TEST_SUITE_P(LayoutsTrianglesTransposesAndDiagonals, Medium_SYCLTpsv,
                                 ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                                    ::testing::Bool(), ::testing::Bool()),
                                 row_name);

        /// @brief Runs tpsv with a stored diagonal that is not one on the host CPU queue.
        class Medium_SYCLTpsvUnitDiagonal : public HostDeviceTest
        {

        };

        /// @brief Tests that tpsv with Diag::Unit never reads the stored diagonal.
        TEST_F(Medium_SYCLTpsvUnitDiagonal, TheStoredDiagonalIsNotRead)
        {
            const int64_t n = 40;
            // The stored diagonal holds small integers other than one; the solve must read ones.
            const std::vector<double> a = small_integers<double>(n * n, 1);
            const std::vector<double> solution = small_integers<double>(n, 2);
            const std::vector<double> packed = to_packed(a, n, Core::Uplo::Upper, Core::Layout::RowMajor);
            const SharedArray<double> device_packed(queue(), packed);
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            SharedArray<double> device_x(
                queue(), dense_matvec(triangle_of(a, n, true, true), n, n, false, 1.0, solution,
                                      0.0, solution));

            Kernels::SYCL::tpsv(queue(), Core::Layout::RowMajor, Core::Uplo::Upper,
                                Core::Transpose::NoTrans, Core::Diag::Unit, n, device_packed.get(),
                                device_x.get(), 1, work.get(), {}).wait();

            EXPECT_EQ(device_x.values(), solution);
        }
    } // namespace Test
} // namespace SushiBLAS
