/****************************************************************/
/* test_asin.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_asin.cpp
 * @brief Tests elementwise().asin against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_AsinTest : public SushiBLASTest {};

TEST_F(Unit_AsinTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 1.0f});
    
    engine->elementwise().asin(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 0.523598f, -0.523598f, 1.570796f});
}
