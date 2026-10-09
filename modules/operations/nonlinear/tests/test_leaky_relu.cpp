/**************************************************************************/
/* test_leaky_relu.cpp                                                    */
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
 * @file test_leaky_relu.cpp
 * @brief Tests NonLinearOps::leaky_relu and leaky_relu_backward against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_LeakyReLUTest : public SushiBLASTest {};

TEST_F(Unit_LeakyReLUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-2.0f, -0.5f, 0.0f, 3.5f});
    
    SushiBLAS::NonLinearOps(*engine).leaky_relu(t, 0.01f);
    engine->execute();
    
    verify_tensor(t, {-0.02f, -0.005f, 0.0f, 3.5f});
}

TEST_F(Unit_LeakyReLUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(x, {-1.0f, -0.5f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    SushiBLAS::NonLinearOps(*engine).leaky_relu_backward(dy, x, dx, 0.01f);
    engine->execute();
    
    verify_tensor(dx, {0.01f, 0.02f, 3.0f, 4.0f});
}
