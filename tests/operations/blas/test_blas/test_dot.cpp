/**************************************************************************/
/* test_dot.cpp                                                           */
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
 * @file test_dot.cpp
 * @brief Tests BLASOps::dot against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_DOTTest : public SushiBLASTest {};

TEST_F(Medium_DOTTest, SimpleDOT) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});
    auto R = engine->create_tensor({1}); // Scalar result tensor

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});
    fill_tensor(R, {0.0f});

    SushiBLAS::BLASOps(recorder()).dot(X, Y, R);
    
    recording->run().value();

    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    verify_tensor(R, {32.0f});
}
