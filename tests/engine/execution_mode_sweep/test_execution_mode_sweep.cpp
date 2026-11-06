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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
        TEST(Medium_ExecutionModeSweep, NamesBothForcedModesOnce)
        {
            EXPECT_EQ(K_BOTH_MODES[0], ExecutionMode::ForceReference);
            EXPECT_EQ(K_BOTH_MODES[1], ExecutionMode::ForceSYCL);
            EXPECT_NE(K_BOTH_MODES[0], K_BOTH_MODES[1]);
        }
    } // namespace Test
} // namespace SushiBLAS
