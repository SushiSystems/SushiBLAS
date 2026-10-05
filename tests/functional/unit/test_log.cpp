/****************************************************************/
/* test_log.cpp                                                 */
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

class Unit_LogTest : public SushiBLASTest {};

TEST_F(Unit_LogTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    engine->elementwise().log(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 0.693147f, 1.098612f, 1.386294f});
}
