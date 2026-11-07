/**************************************************************************/
/* test_sushiblas_log_project.cpp                                         */
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
 * @file test_sushiblas_log_project.cpp
 * @brief Tests that K_SUSHIBLAS_LOG_PROJECT names SushiBLAS and its level variable.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/core/sushiblas_log_project.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_SushiBLASLogProject, TheProjectIsNamedSushiblas)
        {
            EXPECT_EQ(std::string(K_SUSHIBLAS_LOG_PROJECT.name()), "sushiblas");
        }

        TEST(Small_SushiBLASLogProject, TheLevelVariableIsSbLogLevel)
        {
            EXPECT_EQ(std::string(K_SUSHIBLAS_LOG_PROJECT.level_variable()), "SB_LOG_LEVEL");
        }
    } // namespace Test
} // namespace SushiBLAS
