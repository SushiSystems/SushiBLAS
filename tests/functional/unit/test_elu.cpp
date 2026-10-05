/****************************************************************/
/* test_elu.cpp                                                 */
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

class Unit_ELUTest : public SushiBLASTest {};

TEST_F(Unit_ELUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-1.0f, 0.0f, 1.0f, 2.0f});
    
    engine->nonlinear().elu(t, 1.0f);
    engine->execute().wait();
    
    verify_tensor(t, {-0.632121f, 0.0f, 1.0f, 2.0f});
}

TEST_F(Unit_ELUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(x, {-1.0f, 0.0f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().elu_backward(dy, x, dx, 1.0f);
    engine->execute().wait();
    
    verify_tensor(dx, {0.367879f, 1.0f, 1.0f, 1.0f});
}
