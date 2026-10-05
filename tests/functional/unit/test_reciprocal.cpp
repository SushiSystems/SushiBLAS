/****************************************************************/
/* test_reciprocal.cpp                                          */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_reciprocal.cpp
 * @brief Tests elementwise().reciprocal against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_ReciprocalTest : public SushiBLASTest {};

TEST_F(Unit_ReciprocalTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().reciprocal(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 0.50000f, -0.33333f, 2.00000f});
}
