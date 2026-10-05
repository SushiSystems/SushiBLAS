/****************************************************************/
/* test_acos.cpp                                                */
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

class Unit_AcosTest : public SushiBLASTest {};

TEST_F(Unit_AcosTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 1.0f});
    
    engine->elementwise().acos(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.570796f, 1.047197f, 2.094395f, 0.00000f});
}
