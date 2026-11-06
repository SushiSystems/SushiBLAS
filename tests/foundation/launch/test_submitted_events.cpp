/**************************************************************************/
/* test_submitted_events.cpp                                              */
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
 * @file test_submitted_events.cpp
 * @brief Tests that SubmittedEvents keeps every event of a body in order and converts to the last.
 * @author Mustafa Garip
 * @see tests/foundation/launch/README.md
 */

#include <vector>
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/submitted_events.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that an event converts to a value holding exactly that event. */
        TEST(Small_SubmittedEvents, AnEventConvertsToAValueHoldingOneEvent)
        {
            const sycl::event event;

            const Device::SubmittedEvents events = event;

            ASSERT_EQ(events.events().size(), std::size_t{1});
            EXPECT_TRUE(events.events()[0] == event);
        }

        /** @brief Tests that appended events follow the first one in submission order. */
        TEST(Small_SubmittedEvents, AppendKeepsSubmissionOrder)
        {
            const sycl::event first;
            const sycl::event second;
            const sycl::event third;
            const sycl::event fourth;
            Device::SubmittedEvents events(first);
            events.append(second);
            Device::SubmittedEvents later(std::vector<sycl::event>{third, fourth});
            events.append(later);

            const std::vector<sycl::event>& held = events.events();

            ASSERT_EQ(held.size(), std::size_t{4});
            EXPECT_TRUE(held[0] == first);
            EXPECT_TRUE(held[1] == second);
            EXPECT_TRUE(held[2] == third);
            EXPECT_TRUE(held[3] == fourth);
        }

        /** @brief Tests that a value converts to the last event it holds. */
        TEST(Small_SubmittedEvents, AValueConvertsToTheLastEvent)
        {
            const sycl::event first;
            const sycl::event last;
            const Device::SubmittedEvents some(std::vector<sycl::event>{first, last});

            const sycl::event converted = some;

            EXPECT_TRUE(converted == last);
            EXPECT_FALSE(converted == first);
        }

        /** @brief Tests that a value holding no event converts to an event that is already complete. */
        TEST(Small_SubmittedEvents, AnEmptyValueConvertsToACompleteEvent)
        {
            const Device::SubmittedEvents none;

            const sycl::event converted = none;

            EXPECT_EQ(converted.get_info<sycl::info::event::command_execution_status>(),
                      sycl::info::event_command_status::complete);
        }
    } // namespace Test
} // namespace SushiBLAS
