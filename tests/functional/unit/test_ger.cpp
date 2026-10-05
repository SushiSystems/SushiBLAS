/****************************************************************/
/* test_ger.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_ger.cpp
 * @brief Tests blas().ger against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

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
    engine->blas().ger(X, Y, A);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // A = x * y^T = [1; 2] * [1 2 3] = [1 2 3; 2 4 6].
    verify_tensor(A, {1, 2, 3, 2, 4, 6});
}
