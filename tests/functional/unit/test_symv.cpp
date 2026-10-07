/**************************************************************************/
/* test_symv.cpp                                                          */
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
 * @file test_symv.cpp
 * @brief Tests blas().symv against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SYMVTest : public SushiBLASTest {};

TEST_F(Unit_SYMVTest, SimpleSYMV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    // A is symmetric and filled in full: [1 2 3; 2 4 5; 3 5 6].
    fill_tensor(A, {1, 2, 3, 2, 4, 5, 3, 5, 6});
    
    // X:
    fill_tensor(X, {1, 1, 1});

    // Y:
    fill_tensor(Y, {0, 0, 0});

    SB_LOG_INFO("Submitting SYMV.");
    // upper=false -> lower triangle used by symv
    SushiBLAS::BLASOps(*engine).symv(A, X, Y, false);
    
    engine->execute();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Y = A * X = [1+2+3, 2+4+5, 3+5+6] = [6, 11, 14].
    verify_tensor(Y, {6, 11, 14});
}

namespace
{
    /// @brief Holds the value written into the triangle the operation must not read.
    constexpr float K_POISON = 1000.0f;

    /// @brief Holds the input vector every poisoned case multiplies.
    const std::vector<float> K_X = {1, 2, 3};
}

TEST_F(Unit_SYMVTest, LowerIgnoresAPoisonedUpperTriangle)
{
    auto A = engine->create_tensor({3, 3});
    auto X = engine->create_tensor({3});
    auto Y = engine->create_tensor({3});

    // The lower triangle spells S = [2 1 4; 1 3 5; 4 5 7].
    fill_tensor(A, {2, K_POISON, K_POISON, 1, 3, K_POISON, 4, 5, 7});
    fill_tensor(X, K_X);
    fill_tensor(Y, {0, 0, 0});

    SushiBLAS::BLASOps(*engine).symv(A, X, Y, false);
    engine->execute();

    // S * [1 2 3] = [2+2+12, 1+6+15, 4+10+21].
    verify_tensor(Y, {16, 22, 35});
}

TEST_F(Unit_SYMVTest, UpperIgnoresAPoisonedLowerTriangle)
{
    auto A = engine->create_tensor({3, 3});
    auto X = engine->create_tensor({3});
    auto Y = engine->create_tensor({3});

    // The upper triangle spells S = [1 2 3; 2 4 5; 3 5 6].
    fill_tensor(A, {1, 2, 3, K_POISON, 4, 5, K_POISON, K_POISON, 6});
    fill_tensor(X, K_X);
    fill_tensor(Y, {0, 0, 0});

    SushiBLAS::BLASOps(*engine).symv(A, X, Y, true);
    engine->execute();

    // S * [1 2 3] = [1+4+9, 2+8+15, 3+10+18].
    verify_tensor(Y, {14, 25, 31});
}
