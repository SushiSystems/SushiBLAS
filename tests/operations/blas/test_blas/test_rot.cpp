/**************************************************************************/
/* test_rot.cpp                                                           */
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
 * @file test_rot.cpp
 * @brief Tests BLASOps::rot against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_ROTTest : public SushiBLASTest {};

TEST_F(Medium_ROTTest, SimpleROT) 
{
    const int N = 2;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 0});
    fill_tensor(Y, {0, 1});

    // Rotate by 90 degrees: c = 0, s = 1
    float c = 0.0f;
    float s = 1.0f;

    SushiBLAS::BLASOps(*engine).rot(X, Y, c, s);
    
    engine->execute().value();

    // With c = 0 and s = 1, x becomes y and y becomes -x.
    verify_tensor(X, {0, 1});
    verify_tensor(Y, {-1, 0});
}
