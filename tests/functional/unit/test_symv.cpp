/****************************************************************/
/* test_symv.cpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_symv.cpp
 * @brief Tests blas().symv against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SYMVTest : public SushiBLASTest {};

TEST_F(Unit_SYMVTest, SimpleSYMV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    // A is symmetric and filled in full: [1 2 3; 2 4 5; 3 5 6].
    fill_tensor(A, {1, 2, 3, 2, 4, 5, 3, 5, 6});
    
    // X:
    fill_tensor(X, {1, 1, 1});

    // Y:
    fill_tensor(Y, {0, 0, 0});

    SB_LOG_INFO("Submitting SYMV.");
    // upper=false -> lower triangle used by symv
    engine->blas().symv(A, X, Y, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Y = A * X = [1+2+3, 2+4+5, 3+5+6] = [6, 11, 14].
    verify_tensor(Y, {6, 11, 14});
}
