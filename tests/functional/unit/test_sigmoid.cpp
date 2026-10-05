/****************************************************************/
/* test_sigmoid.cpp                                             */
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

class Unit_SigmoidTest : public SushiBLASTest {};

TEST_F(Unit_SigmoidTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 1.0f, -1.0f, 2.0f});
    
    engine->nonlinear().sigmoid(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.5f, 0.731058f, 0.268941f, 0.880797f});
}

TEST_F(Unit_SigmoidTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto y = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(y, {0.5f, 0.731058f, 0.268941f, 0.880797f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().sigmoid_backward(dy, y, dx);
    engine->execute().wait();
    
    verify_tensor(dx, {0.25f, 0.196612f, 0.196612f, 0.104994f});
}
