/**************************************************************************/
/* test_nrm2.cpp                                                          */
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
 * @file test_nrm2.cpp
 * @brief Tests BLASOps::nrm2 against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <cmath>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_NRM2Test : public SushiBLASTest {};

TEST_F(Medium_NRM2Test, SimpleNRM2) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto R = engine->create_tensor({1}); // Scalar result tensor

    fill_tensor(X, {3, 4, 0});
    fill_tensor(R, {0.0f});

    SB_LOG_INFO("Submitting NRM2.");
    SushiBLAS::BLASOps(*engine).nrm2(X, R);
    
    engine->execute().value();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected norm: sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5
    verify_tensor(R, {5.0f});
}
