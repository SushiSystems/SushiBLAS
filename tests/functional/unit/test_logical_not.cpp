/****************************************************************/
/* test_logical_not.cpp                                         */
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

class Unit_Logical_notTest : public SushiBLASTest {};

TEST_F(Unit_Logical_notTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 0.0f, 3.0f, 0.0f});
    engine->logic().logical_not(a, c);
    engine->execute().wait();
    
    verify_tensor(c, {0.00000f, 1.00000f, 0.00000f, 1.00000f});
}
