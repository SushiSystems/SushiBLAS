/**************************************************************************/
/* test_gemm_joint_matrix.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_gemm_joint_matrix.cpp
 * @brief Tests the joint_matrix HALF GEMM against the local-tiled kernel and its fit check.
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
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_joint_matrix.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds one batched HALF product both kernels compute. */
            struct JointMatrixCase
            {
                Core::Layout layout;     /**< Storage order of every operand. */
                Core::Transpose trans_a; /**< How A is read. */
                Core::Transpose trans_b; /**< How B is read. */
                int64_t m;               /**< Rows of C; a multiple of 16. */
                int64_t n;               /**< Columns of C; a multiple of 16. */
                int64_t k;               /**< Depth of the product; a multiple of 16. */
                int64_t batch;           /**< Products in the batch, stored back to back. */
            };

            /** @brief Returns count HALF values cycling through the integers -2 to 2. */
            [[nodiscard]] std::vector<sycl::half> small_integers(std::size_t count,
                                                                 std::size_t salt)
            {
                constexpr std::size_t K_VALUES = 5;
                std::vector<sycl::half> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = sycl::half(static_cast<float>((i * 7 + salt) % K_VALUES) - 2.0f);
                return values;
            }

            /** @brief Returns the bits of a HALF value. */
            [[nodiscard]] uint16_t bits_of(sycl::half value) noexcept
            {
                return sycl::bit_cast<uint16_t>(value);
            }
        } // namespace

        TEST(Small_GemmJointMatrix, TheFitCheckRefusesWhatAFragmentLoadCannotRead)
        {
            Device::Profile profile;
            profile.supports_half_matrix = true;
            const auto at = [](std::uintptr_t address)
            {
                return reinterpret_cast<const sycl::half*>(address);
            };
            const Kernels::OperandAccess rows{32, 1, false};
            const Kernels::OperandAccess columns{1, 32, false};
            const Kernels::OperandAccess odd_lead{1, 36, false};

            using Kernels::SYCL::Detail::joint_matrix_fits;
            EXPECT_TRUE(joint_matrix_fits(profile, 32, 32, 32, at(64), rows, 1024, at(128),
                                          columns, 1024));
            EXPECT_FALSE(joint_matrix_fits(profile, 24, 32, 32, at(64), rows, 1024, at(128),
                                           columns, 1024));
            EXPECT_FALSE(joint_matrix_fits(profile, 32, 32, 40, at(64), rows, 1024, at(128),
                                           columns, 1024));
            EXPECT_FALSE(joint_matrix_fits(profile, 32, 32, 32, at(64), odd_lead, 1024, at(128),
                                           columns, 1024));
            EXPECT_FALSE(joint_matrix_fits(profile, 32, 32, 32, at(66), rows, 1024, at(128),
                                           columns, 1024));
            EXPECT_FALSE(joint_matrix_fits(profile, 32, 32, 32, at(64), rows, 1032, at(128),
                                           columns, 1024));

            profile.supports_half_matrix = false;
            EXPECT_FALSE(joint_matrix_fits(profile, 32, 32, 32, at(64), rows, 1024, at(128),
                                           columns, 1024));
        }

        TEST(Large_GemmJointMatrix, MatchesTheLocalTiledKernelBitForBit)
        {
            sycl::queue& queue = TestEnvironment::queue();
            const Device::Profile& profile = Device::profile_for(queue);
            if (!profile.supports_half_matrix)
                GTEST_SKIP() << "the device offers no HALF joint_matrix product";

            // Small integers sum exactly in FLOAT32, so both kernels round the same value once.
            using Core::Layout;
            using Core::Transpose;
            const std::vector<JointMatrixCase> cases = {
                {Layout::ColumnMajor, Transpose::NoTrans, Transpose::NoTrans, 1024, 1024, 1024, 1},
                {Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, 48, 80, 32, 1},
                {Layout::RowMajor, Transpose::Trans, Transpose::NoTrans, 48, 80, 32, 1},
                {Layout::ColumnMajor, Transpose::NoTrans, Transpose::Trans, 48, 80, 32, 1},
                {Layout::ColumnMajor, Transpose::Trans, Transpose::Trans, 80, 48, 64, 1},
                {Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, 64, 64, 48, 2}};

            for (std::size_t index = 0; index < cases.size(); ++index)
            {
                SCOPED_TRACE("case " + std::to_string(index));
                const JointMatrixCase& c = cases[index];
                const bool row_major = c.layout == Layout::RowMajor;
                const int64_t a_rows = Core::is_transposed(c.trans_a) ? c.k : c.m;
                const int64_t a_cols = Core::is_transposed(c.trans_a) ? c.m : c.k;
                const int64_t b_rows = Core::is_transposed(c.trans_b) ? c.n : c.k;
                const int64_t b_cols = Core::is_transposed(c.trans_b) ? c.k : c.n;
                const int64_t lda = row_major ? a_cols : a_rows;
                const int64_t ldb = row_major ? b_cols : b_rows;
                const int64_t ldc = row_major ? c.n : c.m;
                const int64_t str_a = a_rows * a_cols;
                const int64_t str_b = b_rows * b_cols;
                const int64_t str_c = c.m * c.n;

                SharedArray<sycl::half> a(queue, small_integers(
                                                     static_cast<std::size_t>(c.batch * str_a), 1));
                SharedArray<sycl::half> b(queue, small_integers(
                                                     static_cast<std::size_t>(c.batch * str_b), 3));
                SharedArray<sycl::half> matrix_c(queue, static_cast<std::size_t>(c.batch * str_c));
                SharedArray<sycl::half> tiled_c(queue, static_cast<std::size_t>(c.batch * str_c));

                const Kernels::OperandAccess a_access =
                    Kernels::operand_access(c.layout, c.trans_a, lda);
                const Kernels::OperandAccess b_access =
                    Kernels::operand_access(c.layout, c.trans_b, ldb);
                const Kernels::OperandAccess c_access =
                    Kernels::operand_access(c.layout, Transpose::NoTrans, ldc);
                ASSERT_TRUE(Kernels::SYCL::Detail::joint_matrix_fits(
                    profile, c.m, c.n, c.k, a.get(), a_access, str_a, b.get(), b_access, str_b));

                const sycl::half one(1.0f);
                const sycl::half zero(0.0f);
                Kernels::SYCL::gemm_joint_matrix(
                    queue, c.m, c.n, c.k, one, a.get(), a_access, str_a, b.get(), b_access, str_b,
                    zero, matrix_c.get(), c_access, str_c, c.batch, {},
                    Kernels::GemmIdentityEpilogue{}).wait();
                Kernels::SYCL::Detail::gemm_local_tiled<sycl::half>(
                    queue, c.m, c.n, c.k, one, a.get(), a_access, str_a, b.get(), b_access, str_b,
                    zero, tiled_c.get(), c_access, str_c, c.batch, {},
                    Kernels::GemmIdentityEpilogue{}).wait();

                const std::vector<sycl::half> got = matrix_c.values();
                const std::vector<sycl::half> want = tiled_c.values();
                for (std::size_t i = 0; i < got.size(); ++i)
                    ASSERT_EQ(bits_of(got[i]), bits_of(want[i])) << "element " << i;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
