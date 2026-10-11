/**************************************************************************/
/* test_min.cpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_min.cpp
 * @brief Tests ElementwiseOps::min against fixed expected values.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_MinTest : public SushiBLASTest {};

TEST_F(Unit_MinTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});
    
    SushiBLAS::ElementwiseOps(*engine).min(a, b, c);
    engine->execute().value();
    
    verify_tensor(c, {1.00000f, 0.50000f, -3.00000f, -1.50000f});
}

/**
 * @brief Tests that min returns NaN when either operand is NaN.
 */
TEST_F(Unit_MinTest, NaNInEitherOperandPropagates)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto a = create_initialized_tensor<float>({3}, {nan, 1.0f, 2.0f});
    auto b = create_initialized_tensor<float>({3}, {1.0f, nan, 3.0f});
    auto c = engine->create_tensor({3});

    SushiBLAS::ElementwiseOps(*engine).min(a, b, c);
    engine->execute().value();

    const std::vector<float> got = read_tensor<float>(c);
    EXPECT_TRUE(std::isnan(got[0]));
    EXPECT_TRUE(std::isnan(got[1]));
    EXPECT_FLOAT_EQ(2.0f, got[2]);
}
