/****************************************************************/
/* test_swap.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_swap.cpp
 * @brief Tests blas().swap against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SWAPTest : public SushiBLASTest {};

TEST_F(Unit_SWAPTest, SimpleSWAP) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});

    SB_LOG_INFO("Submitting SWAP.");
    engine->blas().swap(X, Y);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: X = {4, 5, 6}, Y = {1, 2, 3}
    verify_tensor(X, {4, 5, 6});
    verify_tensor(Y, {1, 2, 3});
}
