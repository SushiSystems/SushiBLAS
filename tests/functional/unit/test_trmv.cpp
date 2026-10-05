/****************************************************************/
/* test_trmv.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_trmv.cpp
 * @brief Tests blas().trmv against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_TRMVTest : public SushiBLASTest {};

TEST_F(Unit_TRMVTest, SimpleTRMV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto X = engine->create_tensor({N});

    // A is lower triangular: [1 0 0; 2 4 0; 3 5 6].
    fill_tensor(A, {1, 0, 0, 2, 4, 0, 3, 5, 6});
    
    // X:
    fill_tensor(X, {1, 1, 1});

    SB_LOG_INFO("Submitting TRMV.");
    // upper=false, transA=false, unit_diag=false
    engine->blas().trmv(A, X, false, false, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // X = A * X = [1, 2+4, 3+5+6] = [1, 6, 14].
    verify_tensor(X, {1, 6, 14});
}
