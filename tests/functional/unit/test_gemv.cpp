/**************************************************************************/
/* test_gemv.cpp                                                          */
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
 * @file test_gemv.cpp
 * @brief Tests blas().gemv against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_GEMVTest : public SushiBLASTest {};

TEST_F(Unit_GEMVTest, SimpleGEMV) 
{
    const int M = 2, N = 3;
    auto A = engine->create_tensor({M, N});
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({M});

    // A is [1 2 3; 4 5 6].
    fill_tensor(A, {1, 2, 3, 4, 5, 6});
    
    // X is [1, 1, 1].
    fill_tensor(X, {1, 1, 1});

    // Y is [0, 0].
    fill_tensor(Y, {0, 0});

    SB_LOG_INFO("Submitting GEMV.");
    engine->blas().gemv(A, X, Y);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Y = A * X = [1+2+3, 4+5+6] = [6, 15]
    verify_tensor(Y, {6, 15});
}
