/**************************************************************************/
/* open_collector_fixture.hpp                                             */
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
 * @file open_collector_fixture.hpp
 * @brief Declares the fixture that opens SushiProfiler's process-wide collector for a test.
 * @author Mustafa Garip
 * @see tests/README.md
 */

#pragma once

#include <SushiBLAS/profile/profile_config.hpp>

#if SB_ENABLE_PROFILER

#include <gtest/gtest.h>

#include <SushiProfiler/collector/collector.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Opens the collector for a test and empties it afterwards.
         */
        class OpenCollectorFixture : public ::testing::Test
        {
            protected:
                void SetUp() override
                {
                    close_and_empty();
                    ASSERT_TRUE(SushiProfiler::Collector::instance().open(
                        SushiProfiler::CollectorSettings{}));
                }

                void TearDown() override
                {
                    close_and_empty();
                }

            private:
                /// @brief Closes the collector and discards what its rings hold.
                static void close_and_empty()
                {
                    SushiProfiler::Collector::instance().close();
                    SushiProfiler::DrainBatch discarded;
                    SushiProfiler::Collector::instance().drain(discarded);
                }
        };
    } // namespace Test
} // namespace SushiBLAS

#endif
