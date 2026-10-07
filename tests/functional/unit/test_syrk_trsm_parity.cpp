/**************************************************************************/
/* test_syrk_trsm_parity.cpp                                              */
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
 * @file test_syrk_trsm_parity.cpp
 * @brief Tests SYRK and TRSM SYCL kernels against the reference ones at multi-tile sizes.
 * @author Mustafa Garip
 */

#include <cstdlib>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/blas_flags.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Seed of the random operands of every test in this file.
        constexpr unsigned K_SEED = 20261005u;

        /// @brief Order of the square matrices, large enough to span several tiles.
        constexpr int64_t K_N = 96;

        /// @brief Inner extent of SYRK and column count of the TRSM right-hand side.
        constexpr int64_t K_K = 80;

        /// @brief Absolute tolerance, as the GEMM differential test uses for float and double.
        constexpr double K_TOLERANCE = 1e-4;

        /**
         * @brief Runs one call on the SYCL path and one on the reference path, then compares.
         * @tparam T The element type; float or double.
         */
        template <typename T>
        class Unit_SYRKTRSMParityTest : public SushiBLASTest
        {
            public:
                /** @brief Compares SYRK of a random A on both paths with alpha 1.5 and beta 0.5. */
                void check_syrk(bool upper, bool trans_a)
                {
                    std::srand(K_SEED);
                    const auto dtype = get_sushiblas_dtype<T>();

                    auto A = engine->create_tensor({trans_a ? K_K : K_N, trans_a ? K_N : K_K}, dtype);
                    auto C_actual = engine->create_tensor({K_N, K_N}, dtype);
                    auto C_expected = engine->create_tensor({K_N, K_N}, dtype);

                    fill_random<T>(A);
                    fill_random<T>(C_actual);
                    copy_contents(C_actual, C_expected);

                    engine->set_mode(ExecutionMode::FORCE_SYCL);
                    SushiBLAS::BLASOps(*engine).syrk(A, C_actual, uplo_of(upper),
                                                     transpose_of(trans_a), 1.5, 0.5);
                    engine->execute();

                    engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                    SushiBLAS::BLASOps(*engine).syrk(A, C_expected, uplo_of(upper),
                                                     transpose_of(trans_a), 1.5, 0.5);
                    engine->execute();

                    SB_ASSERT_TENSOR_NEAR(C_actual, C_expected, K_TOLERANCE, T);
                }

                /** @brief Compares TRSM of a diagonally dominant A on both paths with alpha 1.5. */
                void check_trsm(bool left_side, bool upper, bool trans_a, bool unit_diag)
                {
                    std::srand(K_SEED);
                    const auto dtype = get_sushiblas_dtype<T>();

                    auto A = engine->create_tensor({K_N, K_N}, dtype);
                    auto B_actual = engine->create_tensor({left_side ? K_N : K_K,
                                                           left_side ? K_K : K_N}, dtype);
                    auto B_expected = engine->create_tensor({left_side ? K_N : K_K,
                                                             left_side ? K_K : K_N}, dtype);

                    fill_random<T>(A);
                    make_diagonally_dominant(A);
                    fill_random<T>(B_actual);
                    copy_contents(B_actual, B_expected);

                    engine->set_mode(ExecutionMode::FORCE_SYCL);
                    SushiBLAS::BLASOps(*engine).trsm(
                        A, B_actual, side_of(left_side), uplo_of(upper), transpose_of(trans_a),
                        diag_of(unit_diag), 1.5);
                    engine->execute();

                    engine->set_mode(ExecutionMode::FORCE_REFERENCE);
                    SushiBLAS::BLASOps(*engine).trsm(
                        A, B_expected, side_of(left_side), uplo_of(upper), transpose_of(trans_a),
                        diag_of(unit_diag), 1.5);
                    engine->execute();

                    SB_ASSERT_TENSOR_NEAR(B_actual, B_expected, K_TOLERANCE, T);
                }

            private:
                /** @brief Copies every element of @p source into @p target. */
                static void copy_contents(const Tensor& source, Tensor& target)
                {
                    auto& q = TestEnvironment::get_context().get_queue();
                    q.memcpy(target.data(), source.data(),
                             static_cast<std::size_t>(source.num_elements) * sizeof(T)).wait();
                }

                /** @brief Divides every off-diagonal element of @p A by K_N and adds 2 to the diagonal. */
                static void make_diagonally_dominant(Tensor& A)
                {
                    auto& q = TestEnvironment::get_context().get_queue();
                    std::vector<T> host(static_cast<std::size_t>(A.num_elements));
                    q.memcpy(host.data(), A.data(), host.size() * sizeof(T)).wait();

                    for (int64_t i = 0; i < K_N; ++i)
                    {
                        for (int64_t j = 0; j < K_N; ++j)
                        {
                            T& element = host[static_cast<std::size_t>(i * K_N + j)];
                            if (i == j)
                                element += static_cast<T>(2);
                            else
                                element /= static_cast<T>(K_N);
                        }
                    }

                    q.memcpy(A.data(), host.data(), host.size() * sizeof(T)).wait();
                }
        };

        using ParityTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_SYRKTRSMParityTest, ParityTypes);

        TYPED_TEST(Unit_SYRKTRSMParityTest, SyrkLowerNoTranspose)
        {
            this->check_syrk(false, false);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, SyrkUpperNoTranspose)
        {
            this->check_syrk(true, false);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, SyrkLowerTransposed)
        {
            this->check_syrk(false, true);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, SyrkUpperTransposed)
        {
            this->check_syrk(true, true);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, TrsmLeftLowerNoTranspose)
        {
            this->check_trsm(true, false, false, false);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, TrsmLeftUpperTransposed)
        {
            this->check_trsm(true, true, true, false);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, TrsmLeftUpperUnitDiagonal)
        {
            this->check_trsm(true, true, false, true);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, TrsmRightUpperNoTranspose)
        {
            this->check_trsm(false, true, false, false);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, TrsmRightLowerTransposed)
        {
            this->check_trsm(false, false, true, false);
        }

        TYPED_TEST(Unit_SYRKTRSMParityTest, TrsmRightLowerUnitDiagonal)
        {
            this->check_trsm(false, false, false, true);
        }
    } // namespace Test
} // namespace SushiBLAS
