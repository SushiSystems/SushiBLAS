/****************************************************************/
/* test_remainder.cpp                                           */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_RemainderTest : public SushiBLASTest {};

TEST_F(Unit_RemainderTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});
    
    engine->elementwise().remainder(a, b, c);
    engine->execute().wait();
    
    verify_tensor(c, {1.00000f, 0.00000f, 1.00000f, 0.50000f});
}
