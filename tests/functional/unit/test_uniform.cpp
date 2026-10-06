/**************************************************************************/
/* test_uniform.cpp                                                       */
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
 * @file test_uniform.cpp
 * @brief Tests that random().uniform draws inside its bounds.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_UniformTest : public SushiBLASTest {};

TEST_F(Unit_UniformTest, Float32) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    engine->random().uniform(t, 0.0, 1.0);
    engine->execute().wait();
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 0.0f);
        EXPECT_LE(ptr[i], 1.0f);
    }
}
