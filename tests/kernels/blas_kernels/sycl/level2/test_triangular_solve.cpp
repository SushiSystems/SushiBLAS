/**************************************************************************/
/* test_triangular_solve.cpp                                              */
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
 * @file test_triangular_solve.cpp
 * @brief Tests the blocked SYCL triangular solve against the reference one through a packed access.
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
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
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
        /// @brief Names a row after its block width, triangle and transpose.
        static std::string row_name(const ::testing::TestParamInfo<std::tuple<int64_t, bool, bool>>& info)
        {
            return "Block" + std::to_string(std::get<0>(info.param)) +
                   (std::get<1>(info.param) ? "Upper" : "Lower") +
                   (std::get<2>(info.param) ? "Trans" : "NoTrans");
        }

        /// @brief Runs the packed triangular solve on the host CPU queue, one block width, triangle and transpose at a time.
        class Medium_SYCLTriangularSolve : public HostDeviceTest,
                                           public ::testing::WithParamInterface<std::tuple<int64_t, bool, bool>>
        {

        };

        /// @brief Tests that the blocked tpsv recovers the solution the reference tpsv recovers.
        TEST_P(Medium_SYCLTriangularSolve, RecoversTheSolutionTheReferenceRecovers)
        {
            const int64_t n = 70;
            const int64_t block = std::get<0>(GetParam());
            const bool upper = std::get<1>(GetParam());
            const bool transposed = std::get<2>(GetParam());
            // A unit diagonal keeps every substitution exact in integers.
            const std::vector<double> a = unit_diagonal_matrix<double>(n, 1);
            const std::vector<double> solution = small_integers<double>(n, 2);
            SharedArray<Kernels::Accumulator<double>> work(queue(), static_cast<std::size_t>(n));
            const Kernels::SYCL::TriangularSolvePlan plan{block};
            const std::vector<double> packed = to_packed(a, n, uplo_of(upper), Core::Layout::RowMajor);
            const SharedArray<double> device_packed(queue(), packed);
            std::vector<double> x = dense_matvec(triangle_of(a, n, upper, false), n, n, transposed,
                                                 1.0, solution, 0.0, solution);
            SharedArray<double> device_x(queue(), x);

            Kernels::SYCL::tpsv<double>(queue(), plan, Core::Layout::RowMajor, uplo_of(upper),
                                        transpose_of(transposed), Core::Diag::NonUnit, n,
                                        device_packed.get(), device_x.get(), 1, work.get(),
                                        {}).wait();
            Kernels::Reference::tpsv(Core::Layout::RowMajor, uplo_of(upper),
                                     transpose_of(transposed), Core::Diag::NonUnit, n,
                                     packed.data(), x.data(), 1);

            EXPECT_EQ(device_x.values(), x);
            EXPECT_EQ(x, solution);
        }

        INSTANTIATE_TEST_SUITE_P(BlockWidthsTrianglesAndTransposes, Medium_SYCLTriangularSolve,
                                 ::testing::Combine(::testing::Values(int64_t{1}, int64_t{16},
                                                                      int64_t{70}),
                                                    ::testing::Bool(), ::testing::Bool()),
                                 row_name);
    } // namespace Test
} // namespace SushiBLAS
