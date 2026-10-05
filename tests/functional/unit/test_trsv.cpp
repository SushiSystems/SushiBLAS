/****************************************************************/
/* test_trsv.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_trsv.cpp
 * @brief Tests blas().trsv on a 3x3 lower-triangular system.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_TRSVTest : public SushiBLASTest {};

TEST_F(Unit_TRSVTest, SimpleTRSV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto b = engine->create_tensor({N});

    // A is lower triangular: [1 0 0; 2 1 0; 3 4 1].
    fill_tensor(A, {1, 0, 0, 2, 1, 0, 3, 4, 1});
    
    // b: (results from multiplying A by x = [1, 1, 1] which is [1, 3, 8])
    fill_tensor(b, {1, 3, 8});

    SB_LOG_INFO("Submitting TRSV.");
    // upper=false, transA=false, unit_diag=false
    engine->blas().trsv(A, b, false, false, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: b solved to x = [1, 1, 1]
    verify_tensor(b, {1, 1, 1});
}
