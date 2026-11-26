/**************************************************************************/
/* test_gemm_store.cpp                                                    */
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
 * @file test_gemm_store.cpp
 * @brief Tests GemmStore, the store that writes one element of C from its sum.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_store.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Addresses a row-major 2 x 3 matrix.
            const Kernels::OperandAccess K_ROW_MAJOR_2X3{3, 1, false};

            /** @brief Adds ten times the row and the column to a value. */
            struct CoordinateEpilogue
            {
                /** @brief Returns value + 10 * row + col. */
                [[nodiscard]] double operator()(double value, int64_t row,
                                                int64_t col) const noexcept
                {
                    return value + 10.0 * static_cast<double>(row) + static_cast<double>(col);
                }
            };

            /// @brief The store of a GEMM of doubles with no epilogue.
            using IdentityStore =
                Kernels::SYCL::Detail::GemmStore<double, Kernels::GemmIdentityEpilogue>;
        } // namespace

        TEST(Small_GemmStore, AZeroBetaWritesAlphaTimesTheSumOverANaN)
        {
            std::array<double, 6> c{};
            c.fill(std::numeric_limits<double>::quiet_NaN());
            const IdentityStore store(2.0, 0.0, K_ROW_MAJOR_2X3, Kernels::GemmIdentityEpilogue{});

            store(c.data(), 1, 2, 3.5);

            EXPECT_EQ(c[5], 7.0);
        }

        TEST(Small_GemmStore, ANonZeroBetaAddsBetaTimesTheOldValue)
        {
            std::array<double, 6> c{0.0, 0.0, 0.0, 0.0, 4.0, 0.0};
            const IdentityStore store(2.0, -0.5, K_ROW_MAJOR_2X3, Kernels::GemmIdentityEpilogue{});

            store(c.data(), 1, 1, 3.0);

            EXPECT_EQ(c[4], 4.0);
        }

        TEST(Small_GemmStore, TheEpilogueSeesTheFinishedValueAndItsCoordinates)
        {
            std::array<double, 6> c{};
            const Kernels::SYCL::Detail::GemmStore<double, CoordinateEpilogue> store(
                1.0, 0.0, K_ROW_MAJOR_2X3, CoordinateEpilogue{});

            store(c.data(), 1, 2, 0.25);

            EXPECT_EQ(c[5], 12.25);
        }
    } // namespace Test
} // namespace SushiBLAS
