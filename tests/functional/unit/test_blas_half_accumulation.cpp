/**************************************************************************/
/* test_blas_half_accumulation.cpp                                        */
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
 * @file test_blas_half_accumulation.cpp
 * @brief Tests that HALF GEMM, SYRK, GEMV, DOT, ASUM and NRM2 accumulate in FLOAT32 on both
 *        kernels.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Reduction length of every case: 2^12, exact in HALF and twice where HALF sums of ones stall.
        constexpr int64_t K_REDUCTION_LENGTH = 4096;

        /// @brief Edge of the small output dimensions; the reduction length carries the test.
        constexpr int64_t K_OUTPUT_EDGE = 4;

        /// @brief The exact sum of K_REDUCTION_LENGTH products of one and one.
        constexpr double K_EXPECTED_SUM = 4096.0;

        /// @brief Half the HALF spacing at 4096 (spacing 4); a sum stalled at 2048 is far outside it.
        constexpr double K_HALF_ROUNDING_AT_SUM = 2.0;

        /// @brief The exact 2-norm of K_REDUCTION_LENGTH ones: sqrt(4096).
        constexpr double K_EXPECTED_NORM = 64.0;

        /// @brief Half the HALF spacing at 64 (spacing 1/16); a norm stalled at sqrt(2048) is not.
        constexpr double K_HALF_ROUNDING_AT_NORM = 0.03125;

        /**
         * @brief Runs HALF BLAS calls on all-ones operands in a chosen execution mode.
         */
        class Unit_BlasHalfAccumulationTest : public SushiBLASTest
        {
            protected:
                /** @brief Returns a HALF tensor of dims with every element one. */
                [[nodiscard]] Tensor ones(std::initializer_list<int64_t> dims)
                {
                    Tensor t = engine->create_tensor(dims, Core::DataType::HALF);
                    const std::vector<sycl::half> values(static_cast<std::size_t>(t.num_elements),
                                                         sycl::half(1.0f));
                    SushiBLAS::IO(*engine).from_vector(t, values);
                    return t;
                }

                /** @brief Returns a HALF tensor of dims with every element zero. */
                [[nodiscard]] Tensor zeros(std::initializer_list<int64_t> dims)
                {
                    Tensor t = engine->create_tensor(dims, Core::DataType::HALF);
                    const std::vector<sycl::half> values(static_cast<std::size_t>(t.num_elements),
                                                         sycl::half(0.0f));
                    SushiBLAS::IO(*engine).from_vector(t, values);
                    return t;
                }

                /** @brief Returns C = A * B for all-ones A and B, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_gemm(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = ones({K_OUTPUT_EDGE, K_REDUCTION_LENGTH});
                    Tensor B = ones({K_REDUCTION_LENGTH, K_OUTPUT_EDGE});
                    Tensor C = zeros({K_OUTPUT_EDGE, K_OUTPUT_EDGE});
                    SushiBLAS::BLASOps(*engine).gemm(A, B, C, false, false, 1.0, 0.0);
                    engine->execute();
                    return SushiBLAS::IO(*engine).to_vector<sycl::half>(C);
                }

                /** @brief Returns the lower triangle of C = A * A^T for an all-ones A, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_syrk(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = ones({K_OUTPUT_EDGE, K_REDUCTION_LENGTH});
                    Tensor C = zeros({K_OUTPUT_EDGE, K_OUTPUT_EDGE});
                    SushiBLAS::BLASOps(*engine).syrk(A, C, false, false, 1.0, 0.0);
                    engine->execute();

                    const std::vector<sycl::half> full =
                        SushiBLAS::IO(*engine).to_vector<sycl::half>(C);
                    std::vector<sycl::half> lower;
                    for (int64_t i = 0; i < K_OUTPUT_EDGE; ++i)
                        for (int64_t j = 0; j <= i; ++j)
                            lower.push_back(full[static_cast<std::size_t>(i * K_OUTPUT_EDGE + j)]);
                    return lower;
                }

                /** @brief Returns y = A * x for all-ones A and x, computed in mode. */
                [[nodiscard]] std::vector<sycl::half> run_gemv(ExecutionMode mode)
                {
                    engine->set_mode(mode);
                    Tensor A = ones({K_OUTPUT_EDGE, K_REDUCTION_LENGTH});
                    Tensor x = ones({K_REDUCTION_LENGTH});
                    Tensor y = zeros({K_OUTPUT_EDGE});
                    SushiBLAS::BLASOps(*engine).gemv(A, x, y, false, 1.0, 0.0);
                    engine->execute();
                    return SushiBLAS::IO(*engine).to_vector<sycl::half>(y);
                }

                /**
                 * @brief Returns the HALF scalar a Level 1 reduction writes for an all-ones
                 *        vector, computed in mode.
                 * @param call Callable `(Tensor& x, Tensor& result)` that records the reduction.
                 */
                template<typename Call>
                [[nodiscard]] double run_level1(ExecutionMode mode, Call call)
                {
                    engine->set_mode(mode);
                    Tensor x = ones({K_REDUCTION_LENGTH});
                    Tensor result = zeros({1});
                    call(x, result);
                    engine->execute();
                    const sycl::half value =
                        SushiBLAS::IO(*engine).to_vector<sycl::half>(result)[0];
                    return static_cast<double>(static_cast<float>(value));
                }

                /** @brief Expects every element to equal K_EXPECTED_SUM within K_HALF_ROUNDING_AT_SUM. */
                static void expect_full_sum(const std::vector<sycl::half>& actual)
                {
                    ASSERT_FALSE(actual.empty());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_NEAR(static_cast<double>(static_cast<float>(actual[i])),
                                    K_EXPECTED_SUM, K_HALF_ROUNDING_AT_SUM) << "index " << i;
                }

                /** @brief Expects the SYCL result to match the reference result element by element. */
                static void expect_match(const std::vector<sycl::half>& actual,
                                         const std::vector<sycl::half>& expected)
                {
                    ASSERT_EQ(actual.size(), expected.size());
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        EXPECT_NEAR(static_cast<double>(static_cast<float>(actual[i])),
                                    static_cast<double>(static_cast<float>(expected[i])),
                                    K_HALF_ROUNDING_AT_SUM) << "index " << i;
                }
        };

        TEST_F(Unit_BlasHalfAccumulationTest, GemmSumsEveryTermOfALongReduction)
        {
            const auto expected = run_gemm(ExecutionMode::FORCE_REFERENCE);
            const auto actual = run_gemm(ExecutionMode::FORCE_SYCL);
            expect_full_sum(expected);
            expect_full_sum(actual);
            expect_match(actual, expected);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, SyrkSumsEveryTermOfALongReduction)
        {
            const auto expected = run_syrk(ExecutionMode::FORCE_REFERENCE);
            const auto actual = run_syrk(ExecutionMode::FORCE_SYCL);
            expect_full_sum(expected);
            expect_full_sum(actual);
            expect_match(actual, expected);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, GemvSumsEveryTermOfALongReduction)
        {
            const auto expected = run_gemv(ExecutionMode::FORCE_REFERENCE);
            const auto actual = run_gemv(ExecutionMode::FORCE_SYCL);
            expect_full_sum(expected);
            expect_full_sum(actual);
            expect_match(actual, expected);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, DotSumsEveryTermOfALongReduction)
        {
            auto dot = [this](Tensor& x, Tensor& result)
            {
                SushiBLAS::BLASOps(*engine).dot(x, x, result);
            };
            EXPECT_NEAR(run_level1(ExecutionMode::FORCE_REFERENCE, dot), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
            EXPECT_NEAR(run_level1(ExecutionMode::FORCE_SYCL, dot), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, AsumSumsEveryTermOfALongReduction)
        {
            auto asum = [this](Tensor& x, Tensor& result)
            {
                SushiBLAS::BLASOps(*engine).asum(x, result);
            };
            EXPECT_NEAR(run_level1(ExecutionMode::FORCE_REFERENCE, asum), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
            EXPECT_NEAR(run_level1(ExecutionMode::FORCE_SYCL, asum), K_EXPECTED_SUM,
                        K_HALF_ROUNDING_AT_SUM);
        }

        TEST_F(Unit_BlasHalfAccumulationTest, Nrm2SumsEverySquareOfALongReduction)
        {
            auto nrm2 = [this](Tensor& x, Tensor& result)
            {
                SushiBLAS::BLASOps(*engine).nrm2(x, result);
            };
            EXPECT_NEAR(run_level1(ExecutionMode::FORCE_REFERENCE, nrm2), K_EXPECTED_NORM,
                        K_HALF_ROUNDING_AT_NORM);
            EXPECT_NEAR(run_level1(ExecutionMode::FORCE_SYCL, nrm2), K_EXPECTED_NORM,
                        K_HALF_ROUNDING_AT_NORM);
        }
    } // namespace Test
} // namespace SushiBLAS
