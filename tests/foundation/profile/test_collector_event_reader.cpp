/**************************************************************************/
/* test_collector_event_reader.cpp                                        */
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
 * @file test_collector_event_reader.cpp
 * @brief Tests that the collector event reader returns what the calling thread wrote into the collector.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/collector_event_reader.hpp>

#include <gtest/gtest.h>

#include <SushiBLAS/profile/profile.hpp>

#if SB_ENABLE_PROFILER
#include "support/open_collector_fixture.hpp"
#endif

namespace SushiBLAS
{
    namespace Test
    {
#if SB_ENABLE_PROFILER
        using Internal::Profile::CollectorEventReader;

        class Medium_CollectorEventReader : public OpenCollectorFixture
        {
        };

        TEST_F(Medium_CollectorEventReader, ItReadsTheEventsTheCallingThreadWrote)
        {
            {
                SB_ZONE("test_collector_event_reader.zone");
            }
            CollectorEventReader reader;
            EXPECT_EQ(reader.drain_events_of_thread(CollectorEventReader::calling_thread()), 1U);
        }

        TEST_F(Medium_CollectorEventReader, ItReadsNothingForAThreadThatWroteNothing)
        {
            {
                SB_ZONE("test_collector_event_reader.zone");
            }
            CollectorEventReader reader;
            EXPECT_EQ(reader.drain_events_of_thread(CollectorEventReader::calling_thread() + 1U), 0U);
        }
#endif
    } // namespace Test
} // namespace SushiBLAS
