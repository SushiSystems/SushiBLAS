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
#include <SushiBLAS/profile/collector_event_reader.hpp>
#include "support/open_collector_fixture.hpp"
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
        TEST(Small_ProfileGate, TheDefaultLevelOfATranslationUnitIsTwo)
        {
            static_assert(Internal::Profile::K_COMPILED_LEVEL == 2,
                          "the default level of this translation unit");
            EXPECT_EQ(Internal::Profile::K_COMPILED_LEVEL, 2);
        }

        namespace
        {
            /// @brief Opens the collector and reads what a zone wrote through the reader port.
            class Medium_ProfileGate : public OpenCollectorFixture
            {
                protected:
                    /// @brief Returns how many events the calling thread wrote since the last read.
                    std::size_t events_written_by_this_thread()
                    {
                        return m_reader.drain_events_of_thread(
                            Internal::Profile::CollectorEventReader::calling_thread());
                    }

                private:
                    Internal::Profile::CollectorEventReader m_reader;
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
            {
                SB_ZONE_FINE("test_profile_gate.fine");
            }
            EXPECT_EQ(events_written_by_this_thread(), 1U);
        }
#endif
    } // namespace Test
} // namespace SushiBLAS
