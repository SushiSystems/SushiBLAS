/**************************************************************************/
/* test_gemv.cpp                                                          */
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
 * @file test_gemv.cpp
 * @brief Tests BLASOps::gemv against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

class Medium_GEMVTest : public SushiBLASTest {};

TEST_F(Medium_GEMVTest, SimpleGEMV) 
{
    const int M = 2, N = 3;
    auto A = engine->create_tensor({M, N});
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({M});

    // A is [1 2 3; 4 5 6].
    fill_tensor(A, {1, 2, 3, 4, 5, 6});
    
    // X is [1, 1, 1].
    fill_tensor(X, {1, 1, 1});

    // Y is [0, 0].
    fill_tensor(Y, {0, 0});

    SB_LOG_INFO("Submitting GEMV.");
    SushiBLAS::BLASOps(*engine).gemv(A, X, Y);
    
    engine->execute().value();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Y = A * X = [1+2+3, 4+5+6] = [6, 15]
    verify_tensor(Y, {6, 15});
}

/** @brief Tests a transposed product with three long rows of op(A), each read with stride lda. */
TEST_F(Medium_GEMVTest, FewLongStridedRows)
{
    const int M = 20000, N = 3;
    auto A = engine->create_tensor({M, N});
    auto X = engine->create_tensor({M});
    auto Y = engine->create_tensor({N});

    // Column c of A holds c + 1 in every row.
    std::vector<float> a_values(static_cast<std::size_t>(M) * N);
    for (std::size_t i = 0; i < a_values.size(); ++i)
        a_values[i] = static_cast<float>(i % N + 1);
    fill_tensor(A, a_values);
    fill_tensor(X, std::vector<float>(M, 0.5f));
    fill_tensor(Y, {0, 0, 0});

    SushiBLAS::BLASOps(*engine).gemv(A, X, Y, SushiBLAS::Core::Transpose::Trans);
    engine->execute().value();

    verify_tensor(Y, {10000, 20000, 30000});
}
