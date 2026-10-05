/****************************************************************/
/* test_relu.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_relu.cpp
 * @brief Tests nonlinear().relu and relu_backward against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_ReLUTest : public SushiBLASTest {};

TEST_F(Unit_ReLUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-2.0f, -0.5f, 0.0f, 3.5f});
    
    engine->nonlinear().relu(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.0f, 0.0f, 0.0f, 3.5f});
}

TEST_F(Unit_ReLUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(x, {-1.0f, 0.0f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().relu_backward(dy, x, dx);
    engine->execute().wait();
    
    verify_tensor(dx, {0.0f, 0.0f, 3.0f, 4.0f});
}
