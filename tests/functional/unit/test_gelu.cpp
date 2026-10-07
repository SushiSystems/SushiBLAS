/**************************************************************************/
/* test_gelu.cpp                                                          */
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
 * @file test_gelu.cpp
 * @brief Tests nonlinear().gelu and gelu_backward against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_GELUTest : public SushiBLASTest {};

TEST_F(Unit_GELUTest, Forward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {-1.0f, 0.0f, 1.0f, 2.0f});
    
    SushiBLAS::NonLinearOps(*engine).gelu(t);
    engine->execute();
    
    verify_tensor(t, {-0.158865f, 0.0f, 0.841135f, 1.954598f});
}

TEST_F(Unit_GELUTest, Backward) 
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});
    
    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(x, {-1.0f, 0.0f, 1.0f, 2.0f});
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});
    
    SushiBLAS::NonLinearOps(*engine).gelu_backward(dy, x, dx);
    engine->execute();
    
    verify_tensor(dx, {-0.082937f, 0.5f, 1.082937f, 1.086099f});
}
