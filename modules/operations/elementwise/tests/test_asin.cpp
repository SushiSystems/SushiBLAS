/**************************************************************************/
/* test_asin.cpp                                                          */
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
 * @file test_asin.cpp
 * @brief Tests ElementwiseOps::asin against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_AsinTest : public SushiBLASTest {};

TEST_F(Unit_AsinTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 1.0f});
    
    SushiBLAS::ElementwiseOps(*engine).asin(t);
    engine->execute().value();
    
    verify_tensor(t, {0.00000f, 0.523598f, -0.523598f, 1.570796f});
}
