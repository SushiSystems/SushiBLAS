/**************************************************************************/
/* test_blas.cpp                                                          */
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
 * @file test_blas.cpp
 * @brief Tests that the level 3 kernels blas.hpp reaches, the reference one and the SYCL one, give the hand value of a small update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The order of C and the inner dimension of the update.
            constexpr int64_t K_ORDER = 2;

            /// @brief Returns A = [[1, 2], [3, 4]], row by row.
            [[nodiscard]] std::vector<double> operand()
            {
                return {1.0, 2.0, 3.0, 4.0};
            }

            /// @brief Returns the C that syrk leaves on a C of zeros: the upper triangle of A * A^T, which is [[1 + 4, 3 + 8], [., 9 + 16]].
            [[nodiscard]] std::vector<double> expected_c()
            {
                return {5.0, 11.0, 0.0, 25.0};
            }
        } // namespace

        TEST(Small_BlasKernels, TheReferenceLevel3KernelGivesTheHandValue)
        {
            const std::vector<double> a = operand();
            std::vector<double> c(4, 0.0);

            Kernels::Reference::syrk(Core::Layout::RowMajor, Core::Uplo::Upper,
                                     Core::Transpose::NoTrans, K_ORDER, K_ORDER, 1.0, a.data(),
                                     K_ORDER, 0.0, c.data(), K_ORDER);

            EXPECT_EQ(c, expected_c());
        }

        /// @brief Runs the level 3 kernel reached through blas.hpp on the host CPU queue.
        class Medium_BlasKernels : public HostDeviceTest
        {

        };

        TEST_F(Medium_BlasKernels, TheSYCLLevel3KernelGivesTheHandValue)
        {
            const SharedArray<double> a(queue(), operand());
            SharedArray<double> c(queue(), std::vector<double>(4, 0.0));

            Kernels::SYCL::syrk<double>(queue(), Core::Layout::RowMajor, Core::Uplo::Upper,
                                        Core::Transpose::NoTrans, K_ORDER, K_ORDER, 1.0, a.get(),
                                        K_ORDER, 0, 0.0, c.get(), K_ORDER, 0, 1, nullptr, 0,
                                        {}).wait();

            EXPECT_EQ(c.values(), expected_c());
        }
    } // namespace Test
} // namespace SushiBLAS
