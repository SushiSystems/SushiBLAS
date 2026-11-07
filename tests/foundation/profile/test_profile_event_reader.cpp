/**************************************************************************/
/* test_profile_event_reader.cpp                                          */
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
 * @file test_profile_event_reader.cpp
 * @brief Tests that the in-memory event reader returns and discards the events of one thread.
 * @author Mustafa Garip
 * @see tests/foundation/profile/README.md
 */

#include <SushiBLAS/profile/profile_event_reader.hpp>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        using Internal::Profile::InMemoryEventReader;
        using Internal::Profile::ProfileEventReader;

        TEST(Small_ProfileEventReader, AnEmptyReaderReadsNothing)
        {
            InMemoryEventReader reader;
            EXPECT_EQ(reader.drain_events_of_thread(0U), 0U);
        }

        TEST(Small_ProfileEventReader, ItCountsOnlyTheEventsOfTheAskedThread)
        {
            InMemoryEventReader reader;
            reader.add_event(1U);
            reader.add_event(2U);
            reader.add_event(1U);
            EXPECT_EQ(reader.drain_events_of_thread(1U), 2U);
            EXPECT_EQ(reader.drain_events_of_thread(2U), 1U);
        }

        TEST(Small_ProfileEventReader, AReadDiscardsWhatItReturned)
        {
            InMemoryEventReader reader;
            reader.add_event(3U);
            EXPECT_EQ(reader.drain_events_of_thread(3U), 1U);
            EXPECT_EQ(reader.drain_events_of_thread(3U), 0U);
        }

        TEST(Small_ProfileEventReader, ItIsUsableThroughThePort)
        {
            InMemoryEventReader reader;
            reader.add_event(4U);
            ProfileEventReader& port = reader;
            EXPECT_EQ(port.drain_events_of_thread(4U), 1U);
        }
    } // namespace Test
} // namespace SushiBLAS
