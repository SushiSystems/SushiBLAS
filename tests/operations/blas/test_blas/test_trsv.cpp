/**************************************************************************/
/* test_trsv.cpp                                                          */
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
 * @file test_trsv.cpp
 * @brief Tests BLASOps::trsv on a 3x3 lower-triangular system.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/blas_test_oracle/blas_flags.hpp>

using namespace SushiBLAS::Test;

class Unit_TRSVTest : public SushiBLASTest {};

TEST_F(Unit_TRSVTest, SimpleTRSV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto b = engine->create_tensor({N});

    // A is lower triangular: [1 0 0; 2 1 0; 3 4 1].
    fill_tensor(A, {1, 0, 0, 2, 1, 0, 3, 4, 1});
    
    // b: (results from multiplying A by x = [1, 1, 1] which is [1, 3, 8])
    fill_tensor(b, {1, 3, 8});

    SB_LOG_INFO("Submitting TRSV.");
    SushiBLAS::BLASOps(*engine).trsv(A, b, SushiBLAS::Core::Uplo::Lower,
                                     SushiBLAS::Core::Transpose::NoTrans,
                                     SushiBLAS::Core::Diag::NonUnit);
    
    engine->execute().value();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: b solved to x = [1, 1, 1]
    verify_tensor(b, {1, 1, 1});
}

namespace
{
    /// @brief Holds the value written into the triangle the operation must not read.
    constexpr float K_POISON = 1000.0f;

    /// @brief Holds L = [2 0 0; 1 3 0; 4 5 7] with its upper triangle poisoned.
    const std::vector<float> K_LOWER = {2, K_POISON, K_POISON, 1, 3, K_POISON, 4, 5, 7};

    /// @brief Holds U = [1 2 3; 0 4 5; 0 0 6] with its lower triangle poisoned.
    const std::vector<float> K_UPPER = {1, 2, 3, K_POISON, 4, 5, K_POISON, K_POISON, 6};

    /// @brief Holds the solution every poisoned case must recover.
    const std::vector<float> K_SOLUTION = {1, 2, 3};
}

class Unit_TRSVPoisonTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Solves op(A) x = rhs on a 3x3 matrix and expects K_SOLUTION.
         */
        void expect_trsv(const std::vector<float>& matrix, bool upper, bool trans_a,
                         const std::vector<float>& rhs)
        {
            auto A = engine->create_tensor({3, 3});
            auto b = engine->create_tensor({3});
            fill_tensor(A, matrix);
            fill_tensor(b, rhs);

            SushiBLAS::BLASOps(*engine).trsv(A, b, uplo_of(upper), transpose_of(trans_a),
                                             SushiBLAS::Core::Diag::NonUnit);
            engine->execute().value();

            verify_tensor(b, K_SOLUTION);
        }
};

TEST_F(Unit_TRSVPoisonTest, LowerNoTranspose)
{
    // L * [1 2 3] = [2, 7, 35].
    expect_trsv(K_LOWER, false, false, {2, 7, 35});
}

TEST_F(Unit_TRSVPoisonTest, LowerTranspose)
{
    // L^T * [1 2 3] = [16, 21, 21].
    expect_trsv(K_LOWER, false, true, {16, 21, 21});
}

TEST_F(Unit_TRSVPoisonTest, UpperNoTranspose)
{
    // U * [1 2 3] = [14, 23, 18].
    expect_trsv(K_UPPER, true, false, {14, 23, 18});
}

TEST_F(Unit_TRSVPoisonTest, UpperTranspose)
{
    // U^T * [1 2 3] = [1, 10, 31].
    expect_trsv(K_UPPER, true, true, {1, 10, 31});
}
