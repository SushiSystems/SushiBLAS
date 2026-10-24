/**************************************************************************/
/* test_index_options.cpp                                                 */
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

/**
 * @file test_index_options.cpp
 * @brief Tests that IndexOptions names a task and states no determinism class unless told to.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/index/index_options.hpp>
#include <SushiRuntime/core/determinism.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_IndexOptions, ANameAloneStatesNoDeterminismClass)
        {
            const IndexOptions options{"pgs_project"};

            EXPECT_EQ(options.name, "pgs_project");
            EXPECT_EQ(options.determinism, SushiRuntime::Determinism::Unstated);
        }

        TEST(Small_IndexOptions, AClassIsTheRuntimesOwn)
        {
            const IndexOptions options{"pgs_project", SushiRuntime::Determinism::Bitwise};

            EXPECT_EQ(options.determinism, SushiRuntime::Determinism::Bitwise);
        }
    } // namespace Test
} // namespace SushiBLAS
