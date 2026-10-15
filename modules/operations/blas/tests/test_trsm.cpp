/**************************************************************************/
/* test_trsm.cpp                                                          */
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
 * @file test_trsm.cpp
 * @brief Tests BLASOps::trsm in row-major and column-major layouts.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;


class Unit_TRSMTest : public SushiBLASTest {};

TEST_F(Unit_TRSMTest, SimpleTRSM) 
{
    const int M = 2;
    const int N = 2;
    // A must be M x M since we will do left-side multiplication
    auto A = engine->create_tensor({M, M});
    auto B = engine->create_tensor({M, N});

    // A is lower triangular: [2 0; 1 2].
    fill_tensor(A, {2, 0, 1, 2});
    
    // B is [4 6; 4 7].
    fill_tensor(B, {4, 6, 4, 7});

    SB_LOG_INFO("Submitting standard TRSM.");
    SushiBLAS::BLASOps(*engine).trsm(A, B, SushiBLAS::Core::Side::Left,
                                     SushiBLAS::Core::Uplo::Lower,
                                     SushiBLAS::Core::Transpose::NoTrans,
                                     SushiBLAS::Core::Diag::NonUnit, 1.0f);
    
    engine->execute().value();
    SB_LOG_INFO("Execution finished. Checking results.");

    // A * X = B solves to X = [2 3; 1 2], which overwrites B. See tests/unit/README.md.
    verify_tensor(B, {2, 3, 1, 2});
}

TEST_F(Unit_TRSMTest, SimpleTRSMColumnMajor) 
{
    reinit_engine(SushiBLAS::Core::Layout::ColumnMajor);
    
    const int M = 2, N = 2;
    auto A = engine->create_tensor({M, M});
    auto B = engine->create_tensor({M, N});

    // A is upper triangular [2 1; 0 2], stored column-major as [2 0 1 2].
    fill_tensor(A, {2, 0, 1, 2});
    
    // B is [4 6; 4 7], stored column-major as [4 4 6 7].
    fill_tensor(B, {4, 4, 6, 7});

    SB_LOG_INFO("Submitting Column-Major TRSM.");
    SushiBLAS::BLASOps(*engine).trsm(A, B, SushiBLAS::Core::Side::Left,
                                     SushiBLAS::Core::Uplo::Upper,
                                     SushiBLAS::Core::Transpose::NoTrans,
                                     SushiBLAS::Core::Diag::NonUnit, 1.0f);
    
    engine->execute().value();
    SB_LOG_INFO("Execution finished. Checking results.");

    // X is [1 1.25; 2 3.5], stored column-major. See tests/unit/README.md.
    verify_tensor(B, {1.0f, 2.0f, 1.25f, 3.5f});
}
