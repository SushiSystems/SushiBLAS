/**************************************************************************/
/* test_profile_gate.cpp                                                  */
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
 * @file test_profile_gate.cpp
 * @brief Tests that a zone is compiled in or out by the level of its translation unit.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/profile_gate.hpp>

#include <gtest/gtest.h>

#include <SushiBLAS/profile/profile.hpp>

#if SB_ENABLE_PROFILER
#include <vector>
#include <SushiProfiler/collector/collector.hpp>
#include <SushiProfiler/events/trace_event.hpp>
#endif

namespace SushiBLAS
{
    namespace Test
    {
        void write_stage_zone_at_level_one();
        void write_fine_zone_at_level_one();
        int use_every_macro_with_the_profiler_off();

        TEST(Small_ProfileOff, ATranslationUnitCompilesEveryMacroAndCompilesNothingIn)
        {
            EXPECT_EQ(use_every_macro_with_the_profiler_off(), 0);
        }

#if SB_ENABLE_PROFILER
        namespace
        {
            /// @brief Opens the collector for a test and empties it afterwards.
            class Medium_ProfileGate : public ::testing::Test
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

                    /// @brief Returns how many events the calling thread wrote since the last drain.
                    static std::size_t events_written_by_this_thread()
                    {
                        SushiProfiler::DrainBatch batch;
                        SushiProfiler::Collector::instance().drain(batch);
                        for (const SushiProfiler::ThreadDrain& thread : batch.threads)
                        {
                            if (thread.thread_index == SushiProfiler::Collector::thread_index())
                                return thread.events.size();
                        }
                        return 0;
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
        } // namespace

        TEST_F(Medium_ProfileGate, AZoneAtTheCompiledLevelWritesOneEvent)
        {
            write_stage_zone_at_level_one();
            EXPECT_EQ(events_written_by_this_thread(), 1U);
        }

        TEST_F(Medium_ProfileGate, AZoneAboveTheCompiledLevelWritesNothing)
        {
            write_fine_zone_at_level_one();
            EXPECT_EQ(events_written_by_this_thread(), 0U);
        }

        TEST_F(Medium_ProfileGate, TheDefaultLevelCompilesFineZonesIn)
        {
            static_assert(Internal::Profile::K_COMPILED_LEVEL == 2,
                          "the default level of this translation unit");
            {
                SB_ZONE_FINE("test_profile_gate.fine");
            }
            EXPECT_EQ(events_written_by_this_thread(), 1U);
        }
#endif
    } // namespace Test
} // namespace SushiBLAS
