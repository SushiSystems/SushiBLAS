/**************************************************************************/
/* test_syr.cpp                                                           */
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
 * @file test_syr.cpp
 * @brief Tests BLASOps::syr against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_SYRTest : public SushiBLASTest {};

TEST_F(Medium_SYRTest, SimpleSYR) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto A = engine->create_tensor({N, N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(A, std::vector<float>(N * N, 0.0f));

    SushiBLAS::BLASOps(*engine).syr(X, A, SushiBLAS::Core::Uplo::Lower);
    
    engine->execute().value();

    // Only the lower triangle of x*x^T is written. See tests/operations/blas/README.md.
    
    verify_tensor(A, {1, 0, 0, 2, 4, 0, 3, 6, 9});
}
