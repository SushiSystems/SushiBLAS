/**************************************************************************/
/* test_memory_log_port.cpp                                               */
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
 * @file test_memory_log_port.cpp
 * @brief Tests MemoryLogPort and SB_LOG_VIA, which keep records in memory and touch no process-wide state.
 * @author Mustafa Garip
 * @see tests/foundation/core/README.md
 */

#include <cstddef>

#include <gtest/gtest.h>
#include <SushiBLAS/core/memory_log_port.hpp>
#include <SushiBLAS/core/sushiblas_log_project.hpp>
#include <SushiLog/log/log_category.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names the records these tests write, as a module of SushiBLAS would. */
            constexpr SushiLog::LogCategory K_PROBE_LOG_CATEGORY =
                SushiLog::make_log_category(K_SUSHIBLAS_LOG_PROJECT, "probe");
        } // namespace

        TEST(Small_MemoryLogPort, ARecordCarriesTheSushiBLASProjectAndItsModule)
        {
            Core::MemoryLogPort port;

            SB_LOG_VIA(port, SushiLog::LogLevel::Warn, K_PROBE_LOG_CATEGORY, "probe {} of {}", 1, 2);

            ASSERT_EQ(port.records().size(), std::size_t{1});
            EXPECT_EQ(port.records()[0].level, SushiLog::LogLevel::Warn);
            EXPECT_EQ(port.records()[0].project, "sushiblas");
            EXPECT_EQ(port.records()[0].module, "probe");
            EXPECT_EQ(port.records()[0].message, "probe 1 of 2");
        }

        TEST(Small_MemoryLogPort, ARecordBelowThePortLevelIsDropped)
        {
            Core::MemoryLogPort port(SushiLog::LogLevel::Error);

            SB_LOG_VIA(port, SushiLog::LogLevel::Warn, K_PROBE_LOG_CATEGORY, "below the level");

            EXPECT_TRUE(port.records().empty());
        }

        TEST(Small_MemoryLogPort, ARecordAtThePortLevelIsKept)
        {
            Core::MemoryLogPort port(SushiLog::LogLevel::Error);

            SB_LOG_VIA(port, SushiLog::LogLevel::Error, K_PROBE_LOG_CATEGORY, "at the level");

            EXPECT_EQ(port.records().size(), std::size_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
