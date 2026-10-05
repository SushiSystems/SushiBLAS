/****************************************************************/
/* test_silu.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_silu.cpp
 * @brief Tests nonlinear().silu and silu_backward against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SiLUTest : public SushiBLASTest {};

TEST_F(Unit_SiLUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-1.0f, 0.0f, 1.0f, 2.0f});
    
    engine->nonlinear().silu(t);
    engine->execute().wait();
    
    verify_tensor(t, {-0.268941f, 0.0f, 0.731059f, 1.761594f});
}

TEST_F(Unit_SiLUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(x, {-1.0f, 0.0f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().silu_backward(dy, x, dx);
    engine->execute().wait();
    
    verify_tensor(dx, {0.072329f, 0.5f, 0.927671f, 1.090784f});
}
