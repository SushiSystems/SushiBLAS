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
        TEST(Unit_TaskDescription, StartsUnnamedWithEverySlotZero)
        {
            const TaskDescription description;

            EXPECT_EQ(std::strcmp(description.name, "unnamed_task"), 0);
            for (int slot = 0; slot < static_cast<int>(TaskDescription::K_MAX_PARAMETERS); ++slot)
                EXPECT_EQ(description.parameter<std::uint64_t>(slot), std::uint64_t{0});
        }

        TEST(Unit_TaskDescription, ASlotGivesBackTheValueOfItsType)
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
