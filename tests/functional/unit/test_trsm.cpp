/****************************************************************/
/* test_trsm.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_trsm.cpp
 * @brief Tests blas().trsm in row-major and column-major layouts.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

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
    // Left=true, Upper=false, TransA=false, UnitDiag=false, alpha=1.0
    engine->blas().trsm(A, B, true, false, false, false, 1.0f);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // A * X = B solves to X = [2 3; 1 2], which overwrites B. See tests/functional/unit/README.md.
    verify_tensor(B, {2, 3, 1, 2});
}

TEST_F(Unit_TRSMTest, SimpleTRSMColumnMajor) 
{
    reinit_engine(sb::Core::Layout::COLUMN_MAJOR);
    
    const int M = 2, N = 2;
    auto A = engine->create_tensor({M, M});
    auto B = engine->create_tensor({M, N});

    // A is upper triangular [2 1; 0 2], stored column-major as [2 0 1 2].
    fill_tensor(A, {2, 0, 1, 2});
    
    // B is [4 6; 4 7], stored column-major as [4 4 6 7].
    fill_tensor(B, {4, 4, 6, 7});

    SB_LOG_INFO("Submitting Column-Major TRSM.");
    // Left=true, Upper=true, TransA=false, UnitDiag=false, alpha=1.0
    engine->blas().trsm(A, B, true, true, false, false, 1.0f);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // X is [1 1.25; 2 3.5], stored column-major. See tests/functional/unit/README.md.
    verify_tensor(B, {1.0f, 2.0f, 1.25f, 3.5f});
}
