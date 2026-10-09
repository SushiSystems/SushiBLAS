/**************************************************************************/
/* test_any.cpp                                                           */
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
 * @file test_any.cpp
 * @brief Tests LogicOps::any against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_AnyTest : public SushiBLASTest {};

TEST_F(Unit_AnyTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 0.0f, 3.0f, 0.0f});
    auto out = engine->create_tensor({1});
    SushiBLAS::LogicOps(*engine).any(t, out);
    engine->execute();
    
    verify_tensor(out, {1.0f});
}
