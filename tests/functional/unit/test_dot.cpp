/****************************************************************/
/* test_dot.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_dot.cpp
 * @brief Tests blas().dot against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_DOTTest : public SushiBLASTest {};

TEST_F(Unit_DOTTest, SimpleDOT) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});
    auto R = engine->create_tensor({1}); // Scalar result tensor

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});
    fill_tensor(R, {0.0f});

    SB_LOG_INFO("Submitting DOT.");
    engine->blas().dot(X, Y, R);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    verify_tensor(R, {32.0f});
}
