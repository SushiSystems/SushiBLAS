/****************************************************************/
/* test_normal.cpp                                              */
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

class Unit_NormalTest : public SushiBLASTest {};

TEST_F(Unit_NormalTest, Float32) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    engine->random().normal(t, 0.0, 1.0);
    engine->execute().wait();
    EXPECT_EQ(t.num_elements, N);
}
