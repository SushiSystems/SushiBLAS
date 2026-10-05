/****************************************************************/
/* test_gelu.cpp                                                */
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

class Unit_GELUTest : public SushiBLASTest {};

TEST_F(Unit_GELUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-1.0f, 0.0f, 1.0f, 2.0f});
    
    engine->nonlinear().gelu(t);
    engine->execute().wait();
    
    verify_tensor(t, {-0.158865f, 0.0f, 0.841135f, 1.954598f});
}

TEST_F(Unit_GELUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(x, {-1.0f, 0.0f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().gelu_backward(dy, x, dx);
    engine->execute().wait();
    
    verify_tensor(dx, {-0.082937f, 0.5f, 1.082937f, 1.086099f});
}
