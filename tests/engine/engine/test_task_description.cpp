/**************************************************************************/
/* test_task_description.cpp                                              */
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
 * @file test_task_description.cpp
 * @brief Tests the defaults of a TaskDescription and the round trip of its parameter slots.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_TaskDescription, StartsUnnamedWithEverySlotZero)
        {
            const TaskDescription description;

            EXPECT_EQ(std::strcmp(description.name, "unnamed_task"), 0);
            for (int slot = 0; slot < static_cast<int>(TaskDescription::K_MAX_PARAMETERS); ++slot)
                EXPECT_EQ(description.parameter<std::uint64_t>(slot), std::uint64_t{0});
        }

        TEST(Medium_TaskDescription, ASlotGivesBackTheValueOfItsType)
        {
            TaskDescription description;
            description.set_parameter(0, -12);
            description.set_parameter(3, 2.5);
            description.set_parameter(7, 1.25f);

            EXPECT_EQ(description.parameter<int>(0), -12);
            EXPECT_EQ(description.parameter<double>(3), 2.5);
            EXPECT_EQ(description.parameter<float>(7), 1.25f);
        }
    } // namespace Test
} // namespace SushiBLAS
