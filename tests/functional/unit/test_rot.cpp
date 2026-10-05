/****************************************************************/
/* test_rot.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_rot.cpp
 * @brief Tests blas().rot against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_ROTTest : public SushiBLASTest {};

TEST_F(Unit_ROTTest, SimpleROT) 
{
    const int N = 2;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 0});
    fill_tensor(Y, {0, 1});

    // Rotate by 90 degrees: c = 0, s = 1
    float c = 0.0f;
    float s = 1.0f;

    SB_LOG_INFO("Submitting ROT.");
    engine->blas().rot(X, Y, c, s);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // With c = 0 and s = 1, x becomes y and y becomes -x.
    verify_tensor(X, {0, 1});
    verify_tensor(Y, {-1, 0});
}
