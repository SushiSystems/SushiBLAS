/**************************************************************************/
/* test_trsv.cpp                                                          */
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
