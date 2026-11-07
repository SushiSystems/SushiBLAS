/**************************************************************************/
/* test_scoped_log_level.cpp                                              */
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
 * @file test_scoped_log_level.cpp
 * @brief Tests that ScopedLogLevel sets SushiBLAS's level while it lives and puts back the level it found.
 * @author Mustafa Garip
 * @see tests/foundation/test_runner/README.md
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/sushiblas_log_project.hpp>
#include <SushiBLAS/test_runner/scoped_log_level.hpp>
#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_level.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Medium_ScopedLogLevel, SetsTheProjectLevelWhileItLives)
        {
            const ScopedLogLevel level(SushiLog::LogLevel::Trace);

            EXPECT_EQ(SushiLog::log_level(K_SUSHIBLAS_LOG_PROJECT), SushiLog::LogLevel::Trace);
        }

        TEST(Medium_ScopedLogLevel, PutsBackTheLevelItFoundWhenItEnds)
        {
            const SushiLog::LogLevel found = SushiLog::log_level(K_SUSHIBLAS_LOG_PROJECT);
            {
                const ScopedLogLevel level(SushiLog::LogLevel::Trace);
            }

            EXPECT_EQ(SushiLog::log_level(K_SUSHIBLAS_LOG_PROJECT), found);
        }
    } // namespace Test
} // namespace SushiBLAS
