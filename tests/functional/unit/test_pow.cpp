/****************************************************************/
/* test_pow.cpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_pow.cpp
 * @brief Tests elementwise().pow against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_PowTest : public SushiBLASTest {};

TEST_F(Unit_PowTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 0.5f});
    
    engine->elementwise().pow(t, 2.0f);
    engine->execute().wait();
    
    verify_tensor(t, {1.0f, 4.0f, 9.0f, 0.25f});
}
