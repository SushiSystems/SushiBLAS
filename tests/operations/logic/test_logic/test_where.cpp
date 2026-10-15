/**************************************************************************/
/* test_where.cpp                                                         */
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
 * @file test_where.cpp
 * @brief Tests LogicOps::where against fixed expected values.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_WhereTest : public SushiBLASTest {};

TEST_F(Medium_WhereTest, FloatForward) 
{
    auto cond = engine->create_tensor({4});
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(cond, {1.0f, 0.0f, 3.0f, 0.0f}); // act as cond
    fill_tensor(a, {1.0f, 0.0f, 3.0f, 0.0f});
    fill_tensor(b, {10.0f, 20.0f, 30.0f, 40.0f});
    SushiBLAS::LogicOps(*engine).where(cond, a, b, c);
    engine->execute().value();
    
    verify_tensor(c, {1.00000f, 20.00000f, 3.00000f, 40.00000f});
}
