/**************************************************************************/
/* test_trsm_reference.cpp                                                */
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
 * @file test_trsm_reference.cpp
 * @brief Tests the TRSM reference kernel against hand-computed solutions.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Absolute tolerance of every comparison in this file.
        constexpr double K_TOLERANCE = 1e-5;

        /// @brief Fills the triangle the solve must not read.
        constexpr float K_POISON = 999.0f;

        /**
         * @brief Solves 3x3 triangular systems on the reference kernel.
         */
        class Medium_TRSMReferenceTest : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and forces the reference path. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    engine->set_mode(ExecutionMode::ForceReference);
                }

                /** @brief Returns U = [[1,2,3],[0,4,5],[0,0,6]] with its lower part poisoned. */
                [[nodiscard]] static std::vector<float> upper_matrix()
                {
                    return {1.0f, 2.0f, 3.0f,
                            K_POISON, 4.0f, 5.0f,
                            K_POISON, K_POISON, 6.0f};
                }

                /** @brief Returns L = U^T with its upper part poisoned. */
                [[nodiscard]] static std::vector<float> lower_matrix()
                {
                    return {1.0f, K_POISON, K_POISON,
                            2.0f, 4.0f, K_POISON,
                            3.0f, 5.0f, 6.0f};
                }

                /** @brief Returns X = [[1,2],[1,0],[2,-1]], the solution of every left-side case. */
                [[nodiscard]] static std::vector<float> left_solution()
                {
                    return {1.0f, 2.0f, 1.0f, 0.0f, 2.0f, -1.0f};
                }

                /** @brief Returns X = [[1,1,2],[2,0,-1]], the solution of every right-side case. */
                [[nodiscard]] static std::vector<float> right_solution()
                {
                    return {1.0f, 1.0f, 2.0f, 2.0f, 0.0f, -1.0f};
                }

                /**
                 * @brief Solves with a 3x3 A and checks B against @p expected.
                 * @param left_side Makes B 3x2 when true and 2x3 when false.
                 */
                void solve_and_check(const std::vector<float>& a_values,
                                     const std::vector<float>& b_values,
                                     bool left_side, bool upper, bool trans_a, bool unit_diag,
                                     double alpha, const std::vector<float>& expected)
                {
                    const int64_t b_rows = left_side ? 3 : 2;
                    const int64_t b_cols = left_side ? 2 : 3;

                    auto A = create_initialized_tensor<float>({3, 3}, a_values);
                    auto B = engine->create_tensor({b_rows, b_cols}, Core::DataType::Float32);
                    fill_tensor(B, b_values);

                    SushiBLAS::BLASOps(*engine).trsm(
                        A, B, side_of(left_side), uplo_of(upper), transpose_of(trans_a),
                        diag_of(unit_diag), alpha);
                    engine->execute().value();

                    EXPECT_TENSOR_NEAR(B, expected, K_TOLERANCE);
                }
        };

        TEST_F(Medium_TRSMReferenceTest, LeftUpperNoTranspose)
        {
            const std::vector<float> K_B = {9.0f, -1.0f, 14.0f, -5.0f, 12.0f, -6.0f};
            solve_and_check(upper_matrix(), K_B, true, true, false, false, 1.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, LeftUpperNoTransposeScaledByAlpha)
        {
            const std::vector<float> K_B = {4.5f, -0.5f, 7.0f, -2.5f, 6.0f, -3.0f};
            solve_and_check(upper_matrix(), K_B, true, true, false, false, 2.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, LeftUpperTransposed)
        {
            const std::vector<float> K_B = {1.0f, 2.0f, 6.0f, 4.0f, 20.0f, 0.0f};
            solve_and_check(upper_matrix(), K_B, true, true, true, false, 1.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, LeftUpperUnitDiagonal)
        {
            const std::vector<float> K_B = {9.0f, -1.0f, 11.0f, -5.0f, 2.0f, -1.0f};
            solve_and_check(upper_matrix(), K_B, true, true, false, true, 1.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, LeftLowerNoTranspose)
        {
            const std::vector<float> K_B = {1.0f, 2.0f, 6.0f, 4.0f, 20.0f, 0.0f};
            solve_and_check(lower_matrix(), K_B, true, false, false, false, 1.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, LeftLowerTransposed)
        {
            const std::vector<float> K_B = {9.0f, -1.0f, 14.0f, -5.0f, 12.0f, -6.0f};
            solve_and_check(lower_matrix(), K_B, true, false, true, false, 1.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, LeftLowerTransposedUnitDiagonal)
        {
            const std::vector<float> K_B = {9.0f, -1.0f, 11.0f, -5.0f, 2.0f, -1.0f};
            solve_and_check(lower_matrix(), K_B, true, false, true, true, 1.0, left_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, RightUpperNoTranspose)
        {
            const std::vector<float> K_B = {1.0f, 6.0f, 20.0f, 2.0f, 4.0f, 0.0f};
            solve_and_check(upper_matrix(), K_B, false, true, false, false, 1.0, right_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, RightUpperTransposed)
        {
            const std::vector<float> K_B = {9.0f, 14.0f, 12.0f, -1.0f, -5.0f, -6.0f};
            solve_and_check(upper_matrix(), K_B, false, true, true, false, 1.0, right_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, RightUpperUnitDiagonal)
        {
            const std::vector<float> K_B = {1.0f, 3.0f, 10.0f, 2.0f, 4.0f, 5.0f};
            solve_and_check(upper_matrix(), K_B, false, true, false, true, 1.0, right_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, RightLowerNoTranspose)
        {
            const std::vector<float> K_B = {9.0f, 14.0f, 12.0f, -1.0f, -5.0f, -6.0f};
            solve_and_check(lower_matrix(), K_B, false, false, false, false, 1.0, right_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, RightLowerTransposed)
        {
            const std::vector<float> K_B = {1.0f, 6.0f, 20.0f, 2.0f, 4.0f, 0.0f};
            solve_and_check(lower_matrix(), K_B, false, false, true, false, 1.0, right_solution());
        }

        TEST_F(Medium_TRSMReferenceTest, RightLowerTransposedUnitDiagonal)
        {
            const std::vector<float> K_B = {1.0f, 3.0f, 10.0f, 2.0f, 4.0f, 5.0f};
            solve_and_check(lower_matrix(), K_B, false, false, true, true, 1.0, right_solution());
        }
    } // namespace Test
} // namespace SushiBLAS
