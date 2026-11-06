/**************************************************************************/
/* test_ceilings.cpp                                                      */
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
 * @file test_ceilings.cpp
 * @brief Tests that a default DeviceCeilings states nothing a reader would accept as measured.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <gtest/gtest.h>
#include <SushiBLAS/profiling/ceilings.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_Ceilings, ADefaultValueStatesNoDtypeAndNoBandwidth)
        {
            const Profiling::DeviceCeilings ceilings;

            EXPECT_TRUE(ceilings.compute.empty());
            EXPECT_EQ(ceilings.bandwidth.bytes_per_second, 0.0);
            EXPECT_FALSE(ceilings.smoke_only);
        }

        TEST(Small_Ceilings, AComputeCeilingStartsAtZero)
        {
            const Profiling::ComputeCeiling ceiling;

            EXPECT_EQ(ceiling.flops_per_second, 0.0);
            EXPECT_EQ(ceiling.runs, 0U);
        }
    } // namespace Test
} // namespace SushiBLAS
