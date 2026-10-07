/**************************************************************************/
/* test_axpy.cpp                                                          */
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
 * @file test_axpy.cpp
 * @brief Tests BLASOps::axpy against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_AXPYTest : public SushiBLASTest {};

TEST_F(Unit_AXPYTest, SimpleAXPY) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});

    SB_LOG_INFO("Submitting AXPY.");
    SushiBLAS::BLASOps(*engine).axpy(2.0f, X, Y);
    
    engine->execute();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: 2 * X + Y = {2+4, 4+5, 6+6} = {6, 9, 12}
    verify_tensor(Y, {6, 9, 12});
}
