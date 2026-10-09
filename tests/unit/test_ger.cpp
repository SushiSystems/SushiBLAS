/**************************************************************************/
/* test_ger.cpp                                                           */
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
 * @file test_ger.cpp
 * @brief Tests BLASOps::ger against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_GERTest : public SushiBLASTest {};

TEST_F(Unit_GERTest, SimpleGER) 
{
    const int M = 2, N = 3;
    auto X = engine->create_tensor({M});
    auto Y = engine->create_tensor({N});
    auto A = engine->create_tensor({M, N});

    fill_tensor(X, {1, 2});
    fill_tensor(Y, {1, 2, 3});
    
    // Initial A is zeros
    fill_tensor(A, {0, 0, 0, 0, 0, 0});

    SB_LOG_INFO("Submitting GER.");
    SushiBLAS::BLASOps(*engine).ger(X, Y, A);
    
    engine->execute();
    SB_LOG_INFO("Execution finished. Checking results.");

    // A = x * y^T = [1; 2] * [1 2 3] = [1 2 3; 2 4 6].
    verify_tensor(A, {1, 2, 3, 2, 4, 6});
}
