/**************************************************************************/
/* test_execution_mode_sweep.cpp                                          */
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
 * @file test_execution_mode_sweep.cpp
 * @brief Tests that the sweep names each forced execution mode once.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/execution_mode_sweep/execution_mode_sweep.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_ExecutionModeSweep, NamesBothForcedModesOnce)
        {
            EXPECT_EQ(K_BOTH_MODES[0], ExecutionMode::ForceReference);
            EXPECT_EQ(K_BOTH_MODES[1], ExecutionMode::ForceSYCL);
            EXPECT_NE(K_BOTH_MODES[0], K_BOTH_MODES[1]);
        }
    } // namespace Test
} // namespace SushiBLAS
