/****************************************************************/
/* test_leaky_relu.cpp                                          */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_LeakyReLUTest : public SushiBLASTest {};

TEST_F(Unit_LeakyReLUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-2.0f, -0.5f, 0.0f, 3.5f});
    
    engine->nonlinear().leaky_relu(t, 0.01f);
    engine->execute().wait();
    
    verify_tensor(t, {-0.02f, -0.005f, 0.0f, 3.5f});
}

TEST_F(Unit_LeakyReLUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(x, {-1.0f, -0.5f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().leaky_relu_backward(dy, x, dx, 0.01f);
    engine->execute().wait();
    
    verify_tensor(dx, {0.01f, 0.02f, 3.0f, 4.0f});
}
