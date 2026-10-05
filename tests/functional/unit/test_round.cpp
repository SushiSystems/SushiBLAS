/****************************************************************/
/* test_round.cpp                                               */
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

class Unit_RoundTest : public SushiBLASTest {};

TEST_F(Unit_RoundTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.1f, 2.9f, -3.2f, 0.0f});
    
    engine->elementwise().round(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 3.00000f, -3.00000f, 0.00000f});
}
