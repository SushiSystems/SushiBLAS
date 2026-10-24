/**************************************************************************/
/* test_trmv.cpp                                                          */
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
 * @file test_trmv.cpp
 * @brief Tests BLASOps::trmv against fixed expected values.
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

class Medium_TRMVTest : public SushiBLASTest {};

TEST_F(Medium_TRMVTest, SimpleTRMV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto X = engine->create_tensor({N});

    // A is lower triangular: [1 0 0; 2 4 0; 3 5 6].
    fill_tensor(A, {1, 0, 0, 2, 4, 0, 3, 5, 6});
    
    // X:
    fill_tensor(X, {1, 1, 1});

    SushiBLAS::BLASOps(recorder()).trmv(A, X, SushiBLAS::Core::Uplo::Lower,
                                     SushiBLAS::Core::Transpose::NoTrans,
                                     SushiBLAS::Core::Diag::NonUnit);
    
    recording->run().value();

    // X = A * X = [1, 2+4, 3+5+6] = [1, 6, 14].
    verify_tensor(X, {1, 6, 14});
}

namespace
{
    /// @brief Holds the value written into the triangle the operation must not read.
    constexpr float K_POISON = 1000.0f;

    /// @brief Holds L = [2 0 0; 1 3 0; 4 5 7] with its upper triangle poisoned.
    const std::vector<float> K_LOWER = {2, K_POISON, K_POISON, 1, 3, K_POISON, 4, 5, 7};

    /// @brief Holds U = [1 2 3; 0 4 5; 0 0 6] with its lower triangle poisoned.
    const std::vector<float> K_UPPER = {1, 2, 3, K_POISON, 4, 5, K_POISON, K_POISON, 6};

    /// @brief Holds the input vector every poisoned case multiplies.
    const std::vector<float> K_X = {1, 2, 3};
}

class Medium_TRMVPoisonTest : public SushiBLASTest
{
    protected:
        /**
         * @brief Runs trmv on a 3x3 matrix and K_X and checks the product.
         */
        void expect_trmv(const std::vector<float>& matrix, bool upper, bool trans_a,
                         const std::vector<float>& expected)
        {
            auto A = engine->create_tensor({3, 3});
            auto X = engine->create_tensor({3});
            fill_tensor(A, matrix);
            fill_tensor(X, K_X);

            SushiBLAS::BLASOps(recorder()).trmv(A, X, uplo_of(upper), transpose_of(trans_a),
                                             SushiBLAS::Core::Diag::NonUnit);
            recording->run().value();

            verify_tensor(X, expected);
        }
};

TEST_F(Medium_TRMVPoisonTest, LowerNoTranspose)
{
    // L * [1 2 3] = [2, 1+6, 4+10+21].
    expect_trmv(K_LOWER, false, false, {2, 7, 35});
}

TEST_F(Medium_TRMVPoisonTest, LowerTranspose)
{
    // L^T * [1 2 3] = [2+2+12, 6+15, 21].
    expect_trmv(K_LOWER, false, true, {16, 21, 21});
}

TEST_F(Medium_TRMVPoisonTest, UpperNoTranspose)
{
    // U * [1 2 3] = [1+4+9, 8+15, 18].
    expect_trmv(K_UPPER, true, false, {14, 23, 18});
}

TEST_F(Medium_TRMVPoisonTest, UpperTranspose)
{
    // U^T * [1 2 3] = [1, 2+8, 3+10+18].
    expect_trmv(K_UPPER, true, true, {1, 10, 31});
}
