/**************************************************************************/
/* test_sigmoid.cpp                                                       */
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
 * @file test_sigmoid.cpp
 * @brief Tests nonlinear().sigmoid and sigmoid_backward against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SigmoidTest : public SushiBLASTest {};

TEST_F(Unit_SigmoidTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 1.0f, -1.0f, 2.0f});
    
    engine->nonlinear().sigmoid(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.5f, 0.731058f, 0.268941f, 0.880797f});
}

TEST_F(Unit_SigmoidTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto y = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(y, {0.5f, 0.731058f, 0.268941f, 0.880797f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    engine->nonlinear().sigmoid_backward(dy, y, dx);
    engine->execute().wait();
    
    verify_tensor(dx, {0.25f, 0.196612f, 0.196612f, 0.104994f});
}
