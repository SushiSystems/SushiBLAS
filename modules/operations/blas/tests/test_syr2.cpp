/**************************************************************************/
/* test_syr2.cpp                                                          */
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
 * @file test_syr2.cpp
 * @brief Tests BLASOps::syr2 against fixed expected values.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SYR2Test : public SushiBLASTest {};

TEST_F(Unit_SYR2Test, SimpleSYR2) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});
    auto A = engine->create_tensor({N, N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {1, 0, 0}); // Simplified for easy checking
    fill_tensor(A, std::vector<float>(N * N, 0.0f));

    SB_LOG_INFO("Submitting SYR2.");
    SushiBLAS::BLASOps(*engine).syr2(X, Y, A, SushiBLAS::Core::Uplo::Lower);
    
    engine->execute();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Only the lower triangle of x*y^T + y*x^T is written. See tests/unit/README.md.
    
    verify_tensor(A, {2, 0, 0, 2, 0, 0, 3, 0, 0});
}
